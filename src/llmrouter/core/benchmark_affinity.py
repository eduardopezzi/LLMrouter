"""Semantic matching between user prompts and benchmark capability profiles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llmrouter.core.scorer import ScoringResult
from llmrouter.core.types import Tier
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.benchmark_affinity")


class Embedder(Protocol):
    """Minimal embedding interface shared with the semantic role scorer."""

    def encode(self, texts: list[str]) -> list[list[float]] | None:
        """Encode texts into vectors, or return ``None`` when unavailable."""
        ...


@dataclass(frozen=True)
class BenchmarkChunk:
    """One independently embedded semantic view of a benchmark."""

    name: str
    text: str
    tier: Tier
    embedding: list[float]


def load_benchmark_knowledge_base(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the trusted Python knowledge base produced for semantic routing."""
    resolved = Path(path)
    if not resolved.exists():
        _logger.warning("Benchmark knowledge base not found: %s", resolved)
        return {}
    try:
        spec = importlib.util.spec_from_file_location("llmrouter_benchmark_kb", resolved)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = getattr(module, "benchmarks", None)
        if not isinstance(value, dict):
            raise ValueError("knowledge base must define a 'benchmarks' dictionary")
        return {
            str(name): entry
            for name, entry in value.items()
            if isinstance(name, str) and isinstance(entry, dict)
        }
    except Exception as exc:
        _logger.warning("Failed to load benchmark knowledge base %s: %s", resolved, exc)
        return {}


class BenchmarkAffinityScorer:
    """Convert prompt-to-benchmark cosine similarities into normalized weights."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        knowledge_base_path: str | Path,
        embedding_cache_path: str | Path | None = None,
        similarity_threshold: float = 0.30,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._knowledge_base_path = Path(knowledge_base_path)
        self._embedding_cache_path = (
            Path(embedding_cache_path) if embedding_cache_path is not None else None
        )
        self._threshold = max(-1.0, min(1.0, similarity_threshold))
        self._top_k = max(1, top_k)
        self._chunks: list[BenchmarkChunk] = []
        self._loaded = False

    def score(self, prompt: str) -> ScoringResult:
        """Return normalized benchmark affinities in ``ScoringResult.signals``."""
        if not prompt or not prompt.strip():
            return self._fallback("none")
        if not self._ensure_embeddings():
            return self._fallback("unavailable")
        vectors = self._embedder.encode([prompt])
        if not vectors:
            return self._fallback("unavailable")

        from llmrouter.core.semantic_scorer import _cosine_similarity

        per_benchmark: dict[str, tuple[float, Tier]] = {}
        for chunk in self._chunks:
            similarity = _cosine_similarity(vectors[0], chunk.embedding)
            previous = per_benchmark.get(chunk.name)
            if previous is None or similarity > previous[0]:
                per_benchmark[chunk.name] = (similarity, chunk.tier)
        if not per_benchmark:
            return self._fallback("unavailable")

        ranked = sorted(per_benchmark.items(), key=lambda item: item[1][0], reverse=True)
        top_name, (top_similarity, top_tier) = ranked[0]
        eligible = [
            (name, similarity)
            for name, (similarity, _) in ranked[: self._top_k]
            if similarity >= self._threshold
        ]
        if not eligible:
            return ScoringResult(
                score=round(top_similarity, 4),
                tier=Tier.T1,
                signals={
                    "benchmark_top": top_name,
                    "benchmark_confidence": round(top_similarity, 4),
                    "benchmark_affinities": {},
                    "benchmark_used": False,
                },
            )

        raw_weights = {
            name: max(0.01, similarity - self._threshold) for name, similarity in eligible
        }
        total = sum(raw_weights.values())
        weights = {name: round(value / total, 6) for name, value in raw_weights.items()}
        # Correct rounding drift so downstream consumers always receive a unit vector.
        first_name = next(iter(weights))
        weights[first_name] = round(weights[first_name] + (1.0 - sum(weights.values())), 6)

        return ScoringResult(
            score=round(top_similarity, 4),
            tier=top_tier,
            signals={
                "benchmark_top": top_name,
                "benchmark_confidence": round(top_similarity, 4),
                "benchmark_affinities": weights,
                "benchmark_similarities": {
                    name: round(similarity, 4) for name, similarity in eligible
                },
                "benchmark_used": True,
            },
        )

    def _ensure_embeddings(self) -> bool:
        if self._loaded:
            return bool(self._chunks)
        knowledge = load_benchmark_knowledge_base(self._knowledge_base_path)
        if not knowledge:
            self._loaded = True
            return False
        fingerprint = _fingerprint(knowledge)
        cached = self._load_cache(fingerprint)
        if cached is not None:
            self._chunks = cached
            self._loaded = True
            return True

        definitions: list[tuple[str, str, Tier]] = []
        for name, entry in knowledge.items():
            tier = _difficulty_tier(str(entry.get("difficulty", "hard")))
            for text in _benchmark_text_chunks(name, entry):
                definitions.append((name, text, tier))
        vectors = self._embedder.encode([text for _, text, _ in definitions])
        if not vectors or len(vectors) != len(definitions):
            self._loaded = True
            return False
        self._chunks = [
            BenchmarkChunk(name=name, text=text, tier=tier, embedding=vector)
            for (name, text, tier), vector in zip(definitions, vectors, strict=True)
        ]
        self._save_cache(fingerprint)
        self._loaded = True
        return True

    def _load_cache(self, fingerprint: str) -> list[BenchmarkChunk] | None:
        path = self._embedding_cache_path
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
                return None
            return [
                BenchmarkChunk(
                    name=item["name"],
                    text=item["text"],
                    tier=Tier(item["tier"]),
                    embedding=[float(value) for value in item["embedding"]],
                )
                for item in data.get("chunks", [])
            ]
        except Exception as exc:
            _logger.warning("Failed to load benchmark embedding cache: %s", exc)
            return None

    def _save_cache(self, fingerprint: str) -> None:
        path = self._embedding_cache_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "chunks": [
                            {
                                "name": chunk.name,
                                "text": chunk.text,
                                "tier": chunk.tier.value,
                                "embedding": chunk.embedding,
                            }
                            for chunk in self._chunks
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            _logger.warning("Failed to save benchmark embedding cache: %s", exc)

    @staticmethod
    def _fallback(status: str) -> ScoringResult:
        return ScoringResult(
            score=0.0,
            tier=Tier.T1,
            signals={
                "benchmark_top": status,
                "benchmark_confidence": 0.0,
                "benchmark_affinities": {},
                "benchmark_used": False,
            },
        )


def _benchmark_text_chunks(name: str, entry: dict[str, Any]) -> list[str]:
    """Build several bounded semantic views instead of truncating one huge embedding."""
    prefix = f"Benchmark {name}: "
    chunks = [prefix + str(entry.get("description", ""))]
    structured_fields = (
        "domains",
        "skills",
        "reasoning_type",
        "task_types",
        "expected_outputs",
        "evaluation_focus",
        "input_characteristics",
        "output_characteristics",
        "keywords",
        "anti_patterns",
    )
    structured_parts: list[str] = []
    for field_name in structured_fields:
        values = entry.get(field_name, [])
        if isinstance(values, list):
            structured_parts.append(f"{field_name}: {', '.join(map(str, values))}")
    chunks.append(prefix + ". ".join(structured_parts))
    prompts = entry.get("typical_prompts", [])
    if isinstance(prompts, list):
        for index in range(0, len(prompts), 5):
            chunks.append(
                prefix + "Representative requests: " + " ".join(prompts[index : index + 5])
            )
    return [chunk for chunk in chunks if chunk.strip() != prefix.strip()]


def _difficulty_tier(difficulty: str) -> Tier:
    normalized = difficulty.strip().lower()
    if normalized in {"frontier", "expert"}:
        return Tier.T3
    if normalized == "hard":
        return Tier.T2
    return Tier.T1


def _fingerprint(knowledge: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(knowledge, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
