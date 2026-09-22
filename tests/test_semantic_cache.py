"""Tests for the semantic (embedding similarity) response cache."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from llmrouter.core.semantic_cache import SemanticCache
from llmrouter.core.types import ChatResponse, Tier, Usage

_DIM = 64

_PROMPT_A = "What is the capital of France?"
# Same word multiset as _PROMPT_A -> cosine similarity of 1.0, but a
# different raw string -> guaranteed exact-cache miss.
_PROMPT_A_VARIANT = "The capital of France is what?"
_PROMPT_B = "quantum chromodynamics lattice gauge theory"


def _bag_of_words(text: str) -> list[float]:
    """Deterministically map text to a hashed bag-of-words vector."""
    vector = [0.0] * _DIM
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()  # noqa: S324
        vector[int(digest, 16) % _DIM] += 1.0
    return vector


class FakeHashEmbedder:
    """Async embedder: similar texts produce nearby (or identical) vectors."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [_bag_of_words(text) for text in texts]


class SyncEncodeEmbedder:
    """Sync ``encode`` interface — the HybridScorer.embedder protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]] | None:
        self.calls += len(texts)
        return [_bag_of_words(text) for text in texts]


class ExplodingEmbedder:
    """Embedder that always raises — simulates an offline backend."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder offline")


def _response(content: str = "Paris") -> ChatResponse:
    return ChatResponse(
        id="resp-1",
        model="gpt-4o",
        choices=[{"message": {"role": "assistant", "content": content}}],
        usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        latency_ms=50.0,
    )


async def _store_default(
    cache: SemanticCache,
    prompt: str = _PROMPT_A,
    **overrides: Any,
) -> bool:
    """Store one entry with sensible defaults, letting tests override fields."""
    params: dict[str, Any] = {
        "model": "gpt-4o",
        "tier": Tier.T2,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 100,
    }
    params.update(overrides)
    return await cache.store(_response(), prompt, **params)


class TestSemanticLookup:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "semantic_cache.db")

    @pytest.mark.asyncio
    async def test_equivalent_prompts_hit(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder())
        assert await _store_default(cache, _PROMPT_A)

        hit = await cache.lookup_prompt(
            _PROMPT_A_VARIANT,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert hit is not None
        assert hit.choices[0]["message"]["content"] == "Paris"
        assert hit.usage.cache_status == "semantic_hit"
        assert hit.usage.cached_tokens == 7
        assert hit.usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_different_prompts_miss(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder())
        await _store_default(cache, _PROMPT_A)

        result = await cache.lookup_prompt(
            _PROMPT_B,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert result is None
        assert cache.stats()["semantic_misses"] == 1

    @pytest.mark.asyncio
    async def test_threshold_default_allows_near_identical(self, db_path: str) -> None:
        # No embedder needed: raw-embedding lookup path.
        cache = SemanticCache(db_path)
        await _store_default(cache, _PROMPT_A, embedding=[1.0, 0.0, 0.0])

        # cos([1,0,0], [0.976, 0.216, 0]) ~= 0.9764 -> above default 0.95
        hit = await cache.lookup(
            [0.976, 0.216, 0.0],
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert hit is not None

    @pytest.mark.asyncio
    async def test_threshold_configurable_rejects_same_similarity(self, db_path: str) -> None:
        cache = SemanticCache(db_path, threshold=0.99)
        await _store_default(cache, _PROMPT_A, embedding=[1.0, 0.0, 0.0])

        # cos ~= 0.9764 -> below the configured 0.99 threshold
        miss = await cache.lookup(
            [0.976, 0.216, 0.0],
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert miss is None
        assert cache.threshold == 0.99

    @pytest.mark.asyncio
    async def test_zero_vector_embedding_is_miss_not_crash(self, db_path: str) -> None:
        cache = SemanticCache(db_path)
        await _store_default(cache, _PROMPT_A, embedding=[1.0, 0.0, 0.0])

        result = await cache.lookup(
            [0.0, 0.0, 0.0],
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert result is None


class TestSemanticRestrictionFilters:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "semantic_filters.db")

    @pytest.fixture
    async def cache(self, db_path: str) -> SemanticCache:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder())
        await _store_default(cache, _PROMPT_A)
        return cache

    def _lookup(self, cache: SemanticCache, **overrides: Any) -> Any:
        params: dict[str, Any] = {
            "model": "gpt-4o",
            "tier": Tier.T2,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        params.update(overrides)
        return cache.lookup_prompt(_PROMPT_A, **params)

    @pytest.mark.asyncio
    async def test_same_restricted_params_hit(self, cache: SemanticCache) -> None:
        assert await self._lookup(cache) is not None

    @pytest.mark.asyncio
    async def test_different_model_miss(self, cache: SemanticCache) -> None:
        assert await self._lookup(cache, model="claude-3") is None

    @pytest.mark.asyncio
    async def test_different_tier_miss(self, cache: SemanticCache) -> None:
        assert await self._lookup(cache, tier=Tier.T3) is None

    @pytest.mark.asyncio
    async def test_different_temperature_miss(self, cache: SemanticCache) -> None:
        assert await self._lookup(cache, temperature=0.9) is None

    @pytest.mark.asyncio
    async def test_different_top_p_miss(self, cache: SemanticCache) -> None:
        assert await self._lookup(cache, top_p=0.1) is None

    @pytest.mark.asyncio
    async def test_different_max_tokens_miss(self, cache: SemanticCache) -> None:
        assert await self._lookup(cache, max_tokens=200) is None


class TestSemanticPersistenceAndTtl:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "semantic_persist.db")

    @pytest.mark.asyncio
    async def test_reopen_backend_keeps_entries(self, db_path: str) -> None:
        first = SemanticCache(db_path, embedder=FakeHashEmbedder())
        await _store_default(first, _PROMPT_A)

        reopened = SemanticCache(db_path, embedder=FakeHashEmbedder())
        hit = await reopened.lookup_prompt(
            _PROMPT_A_VARIANT,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert hit is not None

    @pytest.mark.asyncio
    async def test_expired_entry_misses(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder())
        await _store_default(cache, _PROMPT_A, ttl_seconds=0.0)

        result = await cache.lookup_prompt(
            _PROMPT_A_VARIANT,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_additive_table_does_not_touch_exact_cache_schema(
        self, tmp_path: Path
    ) -> None:
        from llmrouter.core.cache import SQLiteCacheBackend

        shared_db = str(tmp_path / "shared.db")
        exact = SQLiteCacheBackend(shared_db)
        await exact.set(
            key="k", response={"id": "1"}, model="m", tier=1,
            tokens_total=1, cost_usd=0.0, ttl_seconds=3600.0,
        )

        semantic = SemanticCache(shared_db, embedder=FakeHashEmbedder())
        await _store_default(semantic, _PROMPT_A)

        with sqlite3.connect(shared_db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            exact_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(cache_entries)")
            }
        assert {"cache_entries", "semantic_cache_entries"} <= tables
        assert "embedding_json" not in exact_columns


class TestSemanticFallbacks:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "semantic_fallback.db")

    @pytest.mark.asyncio
    async def test_exploding_embedder_lookup_returns_none_without_raising(
        self, db_path: str
    ) -> None:
        cache = SemanticCache(db_path, embedder=ExplodingEmbedder())

        result = await cache.lookup_prompt(
            _PROMPT_A,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert result is None
        stats = cache.stats()
        assert stats["semantic_unavailable"] == 1
        assert stats["semantic_hits"] == 0
        assert stats["semantic_misses"] == 0

    @pytest.mark.asyncio
    async def test_exploding_embedder_store_returns_false_without_raising(
        self, db_path: str
    ) -> None:
        cache = SemanticCache(db_path, embedder=ExplodingEmbedder())

        stored = await _store_default(cache, _PROMPT_A)

        assert stored is False
        assert cache.stats()["semantic_unavailable"] == 1

    @pytest.mark.asyncio
    async def test_missing_embedder_counts_unavailable(self, db_path: str) -> None:
        cache = SemanticCache(db_path)

        result = await cache.lookup_prompt(
            _PROMPT_A,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert result is None
        assert cache.stats()["semantic_unavailable"] == 1

    @pytest.mark.asyncio
    async def test_sync_encode_embedder_is_supported(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=SyncEncodeEmbedder())
        await _store_default(cache, _PROMPT_A)

        hit = await cache.lookup_prompt(
            _PROMPT_A_VARIANT,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert hit is not None
        assert hit.usage.cache_status == "semantic_hit"


class TestSemanticStats:
    @pytest.mark.asyncio
    async def test_stats_reports_all_counters(self, tmp_path: Path) -> None:
        cache = SemanticCache(str(tmp_path / "stats.db"), embedder=FakeHashEmbedder())
        await _store_default(cache, _PROMPT_A)

        await cache.lookup_prompt(  # hit
            _PROMPT_A_VARIANT,
            model="gpt-4o", tier=Tier.T2, temperature=0.2, top_p=0.9, max_tokens=100,
        )
        await cache.lookup_prompt(  # miss
            _PROMPT_B,
            model="gpt-4o", tier=Tier.T2, temperature=0.2, top_p=0.9, max_tokens=100,
        )

        stats = cache.stats()
        assert stats == {
            "semantic_hits": 1,
            "semantic_misses": 1,
            "semantic_unavailable": 0,
        }
