from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.semantic_scorer import (
    DEFAULT_MODEL_NAME,
    HybridScorer,
    SemanticPromptScorer,
    _cosine_similarity,
    role_from_signals,
)
from llmrouter.core.types import ChatMessage, ChatRequest, ModelInfo, Provider, Tier


class _FakeEmbedder:
    """Produce deterministic embeddings for tests.

    Each role description maps to a distinct one-hot vector. Prompts are
    encoded using the first role word found in lowercase text. Unknown
    prompts get a vector pointing in the opposite direction so cosine
    similarity with every role is negative/zero.
    """

    def __init__(self, role_embeddings: dict[str, list[float]]) -> None:
        self._role_embeddings = dict(role_embeddings)
        dim = len(next(iter(role_embeddings.values()), [0.0]))
        # Unknown vector: all -1.0 orthogonal-ish to one-hot positive vectors.
        self._neutral = [-1.0] * dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector: list[float] | None = None
            for role, emb in self._role_embeddings.items():
                if role in lowered:
                    vector = emb[:]
                    break
            result.append(vector if vector is not None else self._neutral[:])
        return result


def _make_scorer(tmp_path: Path, roles: list[dict[str, object]]) -> SemanticPromptScorer:
    scorer = SemanticPromptScorer(
        model_name=DEFAULT_MODEL_NAME,
        embedding_cache_path=str(tmp_path / "role_embeddings.json"),
        role_definitions=roles,
    )
    # Replace the lazy embedder with our deterministic fake.
    role_embeddings = {
        rd["role"]: [1.0 if i == idx else 0.0 for i in range(len(roles))]
        for idx, rd in enumerate(roles)
    }
    scorer._embedder = _FakeEmbedder(role_embeddings)
    return scorer


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert _cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_semantic_scorer_classifies_role(tmp_path: Path) -> None:
    roles = [
        {"role": "documentation", "description": "documentation", "tier": Tier.T1},
        {"role": "architecture", "description": "architecture", "tier": Tier.T3},
    ]
    scorer = _make_scorer(tmp_path, roles)

    result = scorer.score("write the documentation and README for this module")

    assert result.signals["semantic_role"] == "documentation"
    assert result.tier == Tier.T1
    assert result.signals["semantic_confidence"] > 0


def test_semantic_scorer_downgrades_tier_on_low_confidence(tmp_path: Path) -> None:
    roles = [
        {"role": "architecture", "description": "architecture", "tier": Tier.T3},
    ]
    scorer = _make_scorer(tmp_path, roles)

    # Prompt that does not contain role words → neutral vector → low similarity.
    # Use a truly neutral fake embedder so similarity stays low.
    scorer._roles = [
        role for role in scorer._roles if role.role == "architecture"
    ]
    scorer._embedder = _FakeEmbedder({"architecture": [1.0]})
    result = scorer.score("hello world generic text")

    assert result.signals["semantic_confidence"] < 0.40
    assert result.tier == Tier.T1


def test_semantic_scorer_empty_prompt_returns_t1() -> None:
    scorer = SemanticPromptScorer(role_definitions=[])
    result = scorer.score("   ")
    assert result.tier == Tier.T1
    assert result.score == 0.0


def test_semantic_scorer_caches_embeddings(tmp_path: Path) -> None:
    roles = [
        {"role": "fix", "description": "fix bugs and errors", "tier": Tier.T2},
    ]
    scorer = _make_scorer(tmp_path, roles)
    scorer.score("fix this crash")

    cache_path = tmp_path / "role_embeddings.json"
    assert cache_path.exists()
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert data[0]["role"] == "fix"
    assert len(data[0]["embedding"]) == 1


def test_semantic_scorer_loads_cached_embeddings(tmp_path: Path) -> None:
    roles = [
        {"role": "summarization", "description": "summarization", "tier": Tier.T1},
    ]
    cache_path = tmp_path / "role_embeddings.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "role": "summarization",
                    "description": "summarization",
                    "tier": 1,
                    "embedding": [0.0, 1.0],
                }
            ]
        ),
        encoding="utf-8",
    )

    scorer = SemanticPromptScorer(
        model_name=DEFAULT_MODEL_NAME,
        embedding_cache_path=str(cache_path),
        role_definitions=roles,
    )
    scorer._embedder = _FakeEmbedder({"summarization": [0.0, 1.0]})

    # Call once to trigger cache loading.
    result = scorer.score("summarize the meeting notes")
    assert result.signals["semantic_role"] == "summarization"


def test_hybrid_scorer_blends_signals(tmp_path: Path) -> None:
    roles = [
        {"role": "architecture", "description": "architecture", "tier": Tier.T3},
        {"role": "summarization", "description": "summarization", "tier": Tier.T1},
    ]
    semantic = _make_scorer(tmp_path, roles)
    hybrid = HybridScorer(
        rule_scorer=PromptScorer(),
        semantic_scorer=semantic,
        rule_weight=0.3,
        semantic_weight=0.7,
    )

    result = hybrid.score("design the cloud architecture for our payment service")

    assert result.signals["semantic_role"] == "architecture"
    assert "blended_score" in result.signals
    assert result.signals["semantic_used"] is True


def test_hybrid_scorer_ignores_semantic_when_confidence_low(tmp_path: Path) -> None:
    roles = [
        {"role": "architecture", "description": "architecture", "tier": Tier.T3},
    ]
    semantic = _make_scorer(tmp_path, roles)
    rule = PromptScorer()
    hybrid = HybridScorer(
        rule_scorer=rule,
        semantic_scorer=semantic,
        rule_weight=0.3,
        semantic_weight=0.7,
        semantic_confidence_threshold=0.99,
    )

    # Force the semantic scorer to produce low confidence.
    semantic._embedder = _FakeEmbedder({"architecture": [1.0]})
    semantic._roles = [
        role for role in semantic._roles if role.role == "architecture"
    ]
    result = hybrid.score("hello world generic text")

    assert result.signals["semantic_used"] is False
    assert result.tier == rule.score("hello world generic text").tier


def test_role_from_signals_extracts_role() -> None:
    assert role_from_signals({"semantic_role": "review"}) == "review"
    assert role_from_signals({"other": 1}) is None


def test_multi_model_router_reason_includes_semantic_role(tmp_path: Path) -> None:
    roles = [
        {"role": "documentation", "description": "documentation", "tier": Tier.T1},
        {"role": "architecture", "description": "architecture", "tier": Tier.T3},
    ]
    semantic = _make_scorer(tmp_path, roles)
    hybrid = HybridScorer(semantic_scorer=semantic)

    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="ollama/qwen2.5-coder:3b",
                provider=Provider.OLLAMA,
                tier=Tier.T1,
                capabilities=frozenset({"documentation"}),
                priority=1,
            ),
            ModelInfo(
                name="ollama/deepseek-v4-pro:cloud",
                provider=Provider.OLLAMA,
                tier=Tier.T3,
                capabilities=frozenset({"architecture"}),
                priority=2,
            ),
        )
    )
    import asyncio

    router = MultiModelRouter(registry, hybrid, strategy="quality", fallback_count=1)
    decision = asyncio.run(
        router.route(
            ChatRequest(
                model=None,
                messages=[ChatMessage(role="user", content="Summarize this meeting transcript")],
            )
        )
    )
    assert "role=documentation" in decision.reason or "role=summarization" in decision.reason
    assert decision.primary.tier == Tier.T1


def test_settings_loads_semantic_and_hybrid_config() -> None:
    settings = Settings(
        semantic={
            "enabled": True,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cuda",
        },
        hybrid={"rule_weight": 0.2, "semantic_weight": 0.8, "semantic_confidence_threshold": 0.4},
    )
    assert settings.semantic.enabled is True
    assert settings.semantic.device == "cuda"
    assert settings.hybrid.semantic_weight == 0.8
    assert settings.hybrid.semantic_confidence_threshold == 0.4
