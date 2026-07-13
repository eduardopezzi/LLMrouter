"""Tests for the response cache (Fase 4)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from llmrouter.core.cache import (
    CacheManager,
    CacheStats,
    SQLiteCacheBackend,
    _cache_key,
)
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Tier,
    Usage,
)


class TestCacheKey:
    def test_same_params_same_key(self) -> None:
        k1 = _cache_key("hello world", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("hello world", "gpt-4", 1.0, 1.0, 100)
        assert k1 == k2

    def test_different_prompt_different_key(self) -> None:
        k1 = _cache_key("hello", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("world", "gpt-4", 1.0, 1.0, 100)
        assert k1 != k2

    def test_different_model_different_key(self) -> None:
        k1 = _cache_key("hello", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("hello", "gpt-3.5", 1.0, 1.0, 100)
        assert k1 != k2

    def test_different_temperature_different_key(self) -> None:
        k1 = _cache_key("hello", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("hello", "gpt-4", 0.5, 1.0, 100)
        assert k1 != k2

    def test_different_top_p_different_key(self) -> None:
        k1 = _cache_key("hello", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("hello", "gpt-4", 1.0, 0.9, 100)
        assert k1 != k2

    def test_different_max_tokens_different_key(self) -> None:
        k1 = _cache_key("hello", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("hello", "gpt-4", 1.0, 1.0, 200)
        assert k1 != k2

    def test_none_max_tokens(self) -> None:
        k1 = _cache_key("hello", "gpt-4", 1.0, 1.0, None)
        k2 = _cache_key("hello", "gpt-4", 1.0, 1.0, 0)
        assert k1 == k2  # None normalizes to 0

    def test_prompt_normalized(self) -> None:
        k1 = _cache_key("hello   world", "gpt-4", 1.0, 1.0, 100)
        k2 = _cache_key("hello world", "gpt-4", 1.0, 1.0, 100)
        assert k1 == k2


class TestCacheStats:
    def test_empty(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.tokens_saved == 0
        assert stats.cost_saved_usd == 0.0

    def test_hit_rate(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == 0.3

    def test_to_dict(self) -> None:
        stats = CacheStats(hits=5, misses=5, tokens_saved=100, cost_saved_usd=0.05, entries=2)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["cost_saved_usd"] == 0.05
        assert d["entries"] == 2


class TestSQLiteCacheBackend:
    @pytest.fixture
    def db_path(self) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            yield str(Path(tmp) / "test_cache.db")

    @pytest.mark.asyncio
    async def test_get_miss(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        response = {"id": "123", "choices": [{"message": {"content": "hi"}}]}
        await backend.set(
            key="test-key",
            response=response,
            model="gpt-4",
            tier=Tier.T3,
            tokens_total=10,
            cost_usd=0.001,
            ttl_seconds=3600,
        )
        result = await backend.get("test-key")
        assert result is not None
        assert result["id"] == "123"
        assert result["choices"][0]["message"]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        response = {"id": "expired"}
        await backend.set(
            key="exp-key",
            response=response,
            model="gpt-4",
            tier=Tier.T1,
            tokens_total=5,
            cost_usd=0.0,
            ttl_seconds=0,  # immediately expired
        )
        # Small sleep to ensure time passes
        await asyncio.sleep(0.01)
        result = await backend.get("exp-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_count(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        assert await backend.count() == 0
        await backend.set(
            key="k1", response={"id": "1"}, model="m", tier=1, tokens_total=1, cost_usd=0, ttl_seconds=3600,
        )
        await backend.set(
            key="k2", response={"id": "2"}, model="m", tier=1, tokens_total=1, cost_usd=0, ttl_seconds=3600,
        )
        assert await backend.count() == 2

    @pytest.mark.asyncio
    async def test_persistence(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        await backend.set(
            key="persist", response={"id": "p"}, model="m", tier=1, tokens_total=1, cost_usd=0, ttl_seconds=3600,
        )
        # Create a new backend pointing to the same file
        backend2 = SQLiteCacheBackend(db_path)
        result = await backend2.get("persist")
        assert result is not None
        assert result["id"] == "p"


class TestCacheManager:
    @pytest.fixture
    def db_path(self) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            yield str(Path(tmp) / "test_cache_mgr.db")

    def _make_request(self, prompt: str = "hello", **kwargs: object) -> ChatRequest:
        return ChatRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=float(kwargs.get("temperature", 1.0)),
            top_p=float(kwargs.get("top_p", 1.0)),
            max_tokens=int(kwargs.get("max_tokens", 100) or 0) if kwargs.get("max_tokens") is not None else 100,
        )

    def _make_response(self, content: str = "hi") -> ChatResponse:
        return ChatResponse(
            id="resp-1",
            model="gpt-4",
            choices=[{"message": {"content": content}}],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            latency_ms=100.0,
        )

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        manager = CacheManager(backend)
        req = self._make_request("hello world")
        resp = self._make_response("hi there")

        # First call: miss
        cached = await manager.get(req, "gpt-4", Tier.T3)
        assert cached is None

        # Store response
        await manager.set(req, "gpt-4", Tier.T3, resp, cost_usd=0.001)

        # Second call: hit
        cached = await manager.get(req, "gpt-4", Tier.T3)
        assert cached is not None
        assert cached.choices[0]["message"]["content"] == "hi there"
        assert cached.usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_different_params_miss(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        manager = CacheManager(backend)
        req1 = self._make_request("hello", temperature=1.0)
        req2 = self._make_request("hello", temperature=0.5)
        resp = self._make_response("hi")

        await manager.set(req1, "gpt-4", Tier.T3, resp)
        cached = await manager.get(req2, "gpt-4", Tier.T3)
        assert cached is None  # different temperature

    @pytest.mark.asyncio
    async def test_stats_tracking(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        manager = CacheManager(backend)
        req = self._make_request("stats test")
        resp = self._make_response("ok")

        # Miss
        await manager.get(req, "gpt-4", Tier.T3)
        stats = await manager.stats()
        assert stats.misses == 1
        assert stats.hits == 0

        # Store and hit
        await manager.set(req, "gpt-4", Tier.T3, resp, cost_usd=0.002)
        await manager.get(req, "gpt-4", Tier.T3)
        stats = await manager.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.tokens_saved == 10
        assert stats.cost_saved_usd == 0.002
        assert stats.entries == 1

    @pytest.mark.asyncio
    async def test_ttl_expiry_via_manager(self, db_path: str) -> None:
        backend = SQLiteCacheBackend(db_path)
        # Use a very short TTL for tier 1
        manager = CacheManager(backend, ttl_by_tier={Tier.T1: 0.01})
        req = self._make_request("expire test")
        resp = self._make_response("will expire")

        await manager.set(req, "gpt-4", Tier.T1, resp)
        # Should hit immediately
        cached = await manager.get(req, "gpt-4", Tier.T1)
        assert cached is not None

        # Wait for TTL to expire
        await asyncio.sleep(0.02)
        cached = await manager.get(req, "gpt-4", Tier.T1)
        assert cached is None