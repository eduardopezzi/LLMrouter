"""Semantic prompt scoring with sentence-transformers.

This module provides a drop-in replacement/augmentation for the rule-based
:class:`llmrouter.core.scorer.PromptScorer`. It encodes the incoming prompt
and compares it against pre-computed embeddings of known roles/tasks. The closest
role determines the suggested tier and the semantic confidence score.

If `sentence-transformers` is not installed or the model cannot be loaded, the
scorer returns a transparent fallback result so routing still works.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmrouter.core.scorer import PromptScorer, ScoringResult
from llmrouter.core.types import Tier
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.semantic_scorer")


# Default sentence-transformers model. all-MiniLM-L6-v2 is ~80MB and runs
# comfortably on a 4GB VRAM GPU in CPU mode (default) or CUDA if available.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pre-computed role descriptions and target tiers. These are the canonical
# task roles understood by the LLMrouter catalog.
ROLE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "role": "architecture",
        "description": (
            "Design system architecture, database schemas, infrastructure as code, "
            "cloud diagrams, component relationships, and high-level technical decisions."
        ),
        "tier": Tier.T3,
    },
    {
        "role": "security_audit",
        "description": (
            "Audit code for vulnerabilities, OWASP compliance, injection risks, "
            "secrets leakage, and security best practices."
        ),
        "tier": Tier.T3,
    },
    {
        "role": "review",
        "description": (
            "Perform code review, identify bugs, breaking changes, design flaws, "
            "and suggest improvements to pull requests."
        ),
        "tier": Tier.T3,
    },
    {
        "role": "fix",
        "description": (
            "Generate patches and auto-fix code errors, failing tests, lint issues, "
            "and runtime bugs with human approval."
        ),
        "tier": Tier.T2,
    },
    {
        "role": "refactoring",
        "description": (
            "Clean code, optimize performance, remove legacy patterns, rename symbols, "
            "and restructure code without changing behavior."
        ),
        "tier": Tier.T2,
    },
    {
        "role": "test_generation",
        "description": (
            "Generate unit, integration, e2e tests, mocks, fixtures, and validation "
            "scenarios for codebases."
        ),
        "tier": Tier.T2,
    },
    {
        "role": "migration",
        "description": (
            "Migrate code between languages, frameworks, or library versions; update "
            "deprecated APIs and convert project structures."
        ),
        "tier": Tier.T2,
    },
    {
        "role": "documentation",
        "description": (
            "Write READMEs, docstrings, API contracts, swagger specs, and technical "
            "documentation from code."
        ),
        "tier": Tier.T1,
    },
    {
        "role": "summarization",
        "description": (
            "Summarize commits, pull requests, code diffs, long documents, and chat "
            "histories quickly."
        ),
        "tier": Tier.T1,
    },
]


@dataclass(frozen=True)
class RoleEmbedding:
    """A semantic role with its target tier and optional pre-computed vector."""

    role: str
    description: str
    tier: Tier
    embedding: list[float] | None = None


class _LazyEmbedder:
    """Lazy wrapper around sentence-transformers to keep startup fast."""

    def __init__(self, model_name: str, device: str, cache_dir: str | None) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model: Any | None = None
        self._error: Exception | None = None

    def encode(self, texts: list[str]) -> list[list[float]] | None:
        if self._error is not None:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                _logger.info(
                    "Loading semantic embedding model %s on device %s",
                    self._model_name,
                    self._device,
                )
                self._model = SentenceTransformer(
                    self._model_name, device=self._device, cache_folder=self._cache_dir
                )
            except Exception as exc:  # pragma: no cover - import/model load failures
                _logger.warning("Failed to load sentence-transformers model: %s", exc)
                self._error = exc
                return None
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [list(map(float, embedding)) for embedding in embeddings]
        except Exception as exc:  # pragma: no cover - runtime failures
            _logger.warning("Failed to encode prompt with sentence-transformers: %s", exc)
            self._error = exc
            return None


class SemanticPromptScorer:
    """Semantic scorer that classifies prompts by role using embeddings.

    Implements the same shape as :class:`PromptScorer` so it can be used as a
    drop-in replacement inside :class:`MultiModelRouter`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        cache_dir: str | None = None,
        role_definitions: list[dict[str, Any]] | None = None,
        embedding_cache_path: str | Path | None = None,
    ) -> None:
        self._embedder = _LazyEmbedder(model_name, device, cache_dir)
        self._role_definitions = role_definitions or ROLE_DEFINITIONS
        self._embedding_cache_path = embedding_cache_path
        self._roles: list[RoleEmbedding] = []
        self._loaded = False

    def _ensure_role_embeddings(self) -> bool:
        """Load or compute role embeddings. Returns True if ready."""
        if self._loaded:
            return bool(self._roles)

        cached = self._load_cached_embeddings()
        if cached is not None:
            self._roles = cached
            self._loaded = True
            return True

        descriptions = [rd["description"] for rd in self._role_definitions]
        vectors = self._embedder.encode(descriptions)
        if vectors is None:
            self._loaded = True
            return False

        self._roles = [
            RoleEmbedding(
                role=rd["role"],
                description=rd["description"],
                tier=Tier(rd["tier"]),
                embedding=vector,
            )
            for rd, vector in zip(self._role_definitions, vectors, strict=True)
        ]
        self._save_cached_embeddings()
        self._loaded = True
        return True

    def score(self, prompt: str) -> ScoringResult:
        """Score the prompt semantically and return a ScoringResult."""
        if not prompt or not prompt.strip():
            return ScoringResult(score=0.0, tier=Tier.T1, signals={"semantic_role": "none"})

        if not self._ensure_role_embeddings():
            _logger.warning("Semantic scorer unavailable; returning neutral fallback score")
            return ScoringResult(
                score=0.5,
                tier=Tier.T2,
                signals={
                    "semantic_role": "unknown",
                    "semantic_confidence": 0.0,
                    "fallback": True,
                },
            )

        vectors = self._embedder.encode([prompt])
        if vectors is None or not vectors:
            return ScoringResult(
                score=0.5,
                tier=Tier.T2,
                signals={
                    "semantic_role": "unknown",
                    "semantic_confidence": 0.0,
                    "fallback": True,
                },
            )
        prompt_vector = vectors[0]

        similarities: list[tuple[str, Tier, float]] = []
        for role in self._roles:
            if role.embedding is None:
                continue
            sim = _cosine_similarity(prompt_vector, role.embedding)
            similarities.append((role.role, role.tier, sim))

        if not similarities:
            return ScoringResult(score=0.5, tier=Tier.T2, signals={"semantic_role": "unknown"})

        similarities.sort(key=lambda item: item[2], reverse=True)
        top_role, top_tier, top_sim = similarities[0]
        second_sim = similarities[1][2] if len(similarities) > 1 else 0.0
        margin = top_sim - second_sim

        # Confidence calibration: if top similarity is low, downgrade tier to be safe.
        effective_tier = top_tier
        if top_sim < 0.35:
            effective_tier = Tier.T1
        elif top_sim < 0.50 and top_tier.value > Tier.T1.value:
            effective_tier = Tier(top_tier.value - 1)

        signals = {
            "semantic_role": top_role,
            "semantic_confidence": round(top_sim, 4),
            "semantic_margin": round(margin, 4),
            "semantic_tier": effective_tier.value,
        }

        return ScoringResult(
            score=round(top_sim, 4),
            tier=effective_tier,
            signals=signals,
        )

    def _load_cached_embeddings(self) -> list[RoleEmbedding] | None:
        if self._embedding_cache_path is None:
            return None
        path = Path(self._embedding_cache_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [
                RoleEmbedding(
                    role=item["role"],
                    description=item["description"],
                    tier=Tier(item["tier"]),
                    embedding=item["embedding"],
                )
                for item in data
            ]
        except Exception as exc:  # pragma: no cover - cache read failures
            _logger.warning("Failed to load semantic embedding cache: %s", exc)
            return None

    def _save_cached_embeddings(self) -> None:
        if self._embedding_cache_path is None or not self._roles:
            return
        path = Path(self._embedding_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "role": role.role,
                "description": role.description,
                "tier": role.tier.value,
                "embedding": role.embedding,
            }
            for role in self._roles
        ]
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - cache write failures
            _logger.warning("Failed to save semantic embedding cache: %s", exc)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridScorer:
    """Combine rule-based and semantic scorers.

    The final score is a weighted blend of the rule-based complexity score and
    the semantic confidence. The tier is chosen from the highest of the two tier
    recommendations unless the semantic score is very low, in which case only
    the rule-based signal is trusted.
    """

    def __init__(
        self,
        rule_scorer: PromptScorer | None = None,
        semantic_scorer: SemanticPromptScorer | None = None,
        rule_weight: float = 0.3,
        semantic_weight: float = 0.7,
        semantic_confidence_threshold: float = 0.35,
    ) -> None:
        self._rule_scorer = rule_scorer or PromptScorer()
        self._semantic_scorer = semantic_scorer or SemanticPromptScorer()
        total = rule_weight + semantic_weight
        self._rule_weight = rule_weight / total
        self._semantic_weight = semantic_weight / total
        self._semantic_threshold = semantic_confidence_threshold

    def score(self, prompt: str) -> ScoringResult:
        """Score the prompt using both rule-based and semantic signals."""
        rule_result = self._rule_scorer.score(prompt)
        semantic_result = self._semantic_scorer.score(prompt)

        semantic_confidence = semantic_result.signals.get("semantic_confidence", 0.0)
        use_semantic = semantic_confidence >= self._semantic_threshold

        if use_semantic:
            blended_score = min(
                1.0,
                self._rule_weight * rule_result.score
                + self._semantic_weight * semantic_confidence,
            )
            # Choose the higher tier (more conservative) between rule and semantic.
            final_tier = max(rule_result.tier, semantic_result.tier, key=lambda t: t.value)
        else:
            blended_score = rule_result.score
            final_tier = rule_result.tier

        signals: dict[str, Any] = {
            **rule_result.signals,
            **semantic_result.signals,
            "blended_score": round(blended_score, 4),
            "rule_weight": round(self._rule_weight, 4),
            "semantic_weight": round(self._semantic_weight, 4),
            "semantic_used": use_semantic,
        }

        return ScoringResult(score=round(blended_score, 4), tier=final_tier, signals=signals)


def role_from_signals(signals: dict[str, Any]) -> str | None:
    """Convenience helper to extract the inferred semantic role."""
    return str(signals.get("semantic_role")) if "semantic_role" in signals else None
