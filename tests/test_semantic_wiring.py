"""Tests for the semantic cache wiring (A3): config defaults, runtime builder,
embedder adapter (timeout + circuit breaker), create_app injection, cache/stats
endpoint and contract rollout endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmrouter.api.routes import create_app
from llmrouter.config import Settings
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.semantic_cache import SemanticCache
from llmrouter.core.types import ModelInfo, Provider, Tier
from llmrouter.runtime import (
    _build_semantic_cache,
    build_app,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSyncEmbedder:
    """Synchronous embedder mapping texts to deterministic small vectors."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 8
            for index, char in enumerate(text):
                vector[index % 8] += (ord(char) % 17) / 17.0
            norm = sum(value * value for value in vector) or 1.0
            vector = [value / norm**0.5 for value in vector]
            vectors.append(vector)
        return vectors


class _ExplodingSyncEmbedder:
    """Synchronous embedder that always raises."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise RuntimeError("embedder down")


class _SleepingSyncEmbedder:
    """Synchronous embedder that sleeps longer than the configured timeout."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        time.sleep(self._seconds)
        return [[1.0] * 8 for _ in texts]



class _ScorerStub:
    """Duck-typed HybridScorer stand-in exposing ._semantic_scorer.embedder."""

    def __init__(self, embedder: object) -> None:
        self._semantic_scorer = type("_S", (), {"embedder": embedder})()

# ---------------------------------------------------------------------------
# Config defaults (opt-in)
# ---------------------------------------------------------------------------


def test_semantic_cache_config_defaults_are_opt_in() -> None:
    settings = Settings()

    assert settings.semantic_cache.enabled is False
    assert settings.semantic_cache.threshold == pytest.approx(0.95)
    assert settings.semantic_cache.ttl_seconds == 3600
    assert settings.semantic_cache.embed_timeout_seconds == pytest.approx(5.0)
    assert settings.semantic_cache.background_store is True


# ---------------------------------------------------------------------------
# _build_semantic_cache
# ---------------------------------------------------------------------------


def test_build_semantic_cache_returns_none_when_disabled(tmp_path: Path) -> None:
    settings = Settings()
    settings.semantic_cache.db_path = str(tmp_path / "semantic.db")

    assert _build_semantic_cache(settings, scorer=_ScorerStub(None)) is None


def test_build_semantic_cache_builds_cache_with_adapter(tmp_path: Path) -> None:
    settings = Settings()
    settings.semantic_cache.enabled = True
    settings.semantic_cache.db_path = str(tmp_path / "semantic.db")

    cache = _build_semantic_cache(settings, scorer=_ScorerStub(_FakeSyncEmbedder()))

    assert isinstance(cache, SemanticCache)


# ---------------------------------------------------------------------------
# Adapter: to_thread + timeout + circuit breaker
# ---------------------------------------------------------------------------


def test_adapter_embeds_via_to_thread(tmp_path: Path) -> None:
    settings = Settings()
    settings.semantic_cache.enabled = True
    settings.semantic_cache.db_path = str(tmp_path / "semantic.db")

    cache = _build_semantic_cache(settings, scorer=_ScorerStub(_FakeSyncEmbedder()))
    assert cache is not None and cache._embedder is not None

    async def scenario() -> None:
        vectors = await cache._embedder.embed(["hello world"])  # type: ignore[attr-defined]
        assert len(vectors) == 1
        assert len(vectors[0]) > 0

    asyncio.run(scenario())


def test_adapter_timeout_returns_none_fast(tmp_path: Path) -> None:
    settings = Settings()
    settings.semantic_cache.enabled = True
    settings.semantic_cache.db_path = str(tmp_path / "semantic.db")
    settings.semantic_cache.embed_timeout_seconds = 0.05

    cache = _build_semantic_cache(settings, scorer=_ScorerStub(_SleepingSyncEmbedder(2.0)))
    assert cache is not None

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        # lookup_prompt must never raise and must not block for 2s.
        result = await asyncio.wait_for(
            cache.lookup_prompt(
                "some prompt",
                model="m",
                tier=1,
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            ),
            timeout=1.0,
        )
        elapsed = loop.time() - start
        assert result is None
        assert elapsed < 1.0

    asyncio.run(scenario())


def test_circuit_breaker_opens_after_three_failures(tmp_path: Path) -> None:
    settings = Settings()
    settings.semantic_cache.enabled = True
    settings.semantic_cache.db_path = str(tmp_path / "semantic.db")

    embedder = _ExplodingSyncEmbedder()
    cache = _build_semantic_cache(settings, scorer=_ScorerStub(embedder))
    assert cache is not None and cache._embedder is not None
    adapter = cache._embedder

    async def scenario() -> None:
        for _ in range(3):
            result = await cache.lookup_prompt(
                "prompt",
                model="m",
                tier=1,
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
            assert result is None
        assert adapter.consecutive_failures == 3
        assert adapter.circuit_open is True

        # Circuit open: embedder is no longer called.
        calls_before = embedder.calls
        for _ in range(2):
            result = await cache.lookup_prompt(
                "prompt",
                model="m",
                tier=1,
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
            assert result is None
        assert embedder.calls == calls_before

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# create_app wiring + cache/stats endpoint
# ---------------------------------------------------------------------------


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(name="summary", provider=Provider.OPENAI, tier=Tier.T1),
        )
    )


class _NullProxy(ProviderProxy):
    """Proxy stub that never gets called by these tests."""


def test_create_app_injects_semantic_cache_into_state(tmp_path: Path) -> None:
    cache = SemanticCache(str(tmp_path / "semantic.db"))
    app = create_app(registry=_registry(), semantic_cache=cache)

    assert app.state.semantic_cache is cache


def test_cache_stats_exposes_semantic_counters(tmp_path: Path) -> None:
    cache = SemanticCache(str(tmp_path / "semantic.db"))
    app = create_app(registry=_registry(), semantic_cache=cache)

    class _StatsCacheManager:
        async def stats(self):  # noqa: ANN202 - minimal stub
            from llmrouter.core.cache import CacheStats

            return CacheStats(hits=1, misses=2)

    app.state.cache_manager = _StatsCacheManager()
    client = TestClient(app)
    response = client.get("/v1/llmrouter/cache/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["hits"] == 1
    assert body["semantic_hits"] == 0
    assert body["semantic_misses"] == 0
    assert body["semantic_unavailable"] == 0


# ---------------------------------------------------------------------------
# Contract: rollout endpoints present in the snapshot
# ---------------------------------------------------------------------------


def test_contract_snapshot_includes_rollout_endpoints() -> None:
    snapshot = Path("contracts/llmrouter.contract.json")
    data = json.loads(snapshot.read_text(encoding="utf-8"))

    paths = {endpoint["path"] for endpoint in data["endpoints"]}
    assert "/v1/llmrouter/rollout" in paths
    assert "/v1/llmrouter/rollout/{model_name}" in paths


def test_contract_snapshot_includes_cache_verify_endpoint() -> None:
    snapshot = Path("contracts/llmrouter.contract.json")
    data = json.loads(snapshot.read_text(encoding="utf-8"))

    endpoints = [
        endpoint
        for endpoint in data["endpoints"]
        if endpoint["path"] == "/v1/llmrouter/cache/verify"
    ]
    assert len(endpoints) == 1
    endpoint = endpoints[0]
    assert endpoint["auth_required"] is True
    assert endpoint["method"] == "POST"
    assert endpoint["response_schema"] == {
        "buckets": "object",
        "checked": "int",
        "error": "int",
        "mismatch": "int",
        "ok": "int",
    }


def test_contract_snapshot_includes_stream_stats_block() -> None:
    """E2 (S1, Dev E2-A): ``/v1/llmrouter/cache/stats`` exposes stream counters.

    The streaming cache replay (PRD §4.4) introduces ten in-memory counters
    that the route-layer wiring surfaces as a nested ``stream`` object.  The
    contract snapshot must keep this block in sync — this test pins the
    shape so a hand-edit of the contract cannot drop the new fields
    accidentally.
    """
    snapshot = Path("contracts/llmrouter.contract.json")
    data = json.loads(snapshot.read_text(encoding="utf-8"))

    chat_endpoint = next(
        endpoint
        for endpoint in data["endpoints"]
        if endpoint["path"] == "/v1/chat/completions"
    )
    assert "stream_options" in chat_endpoint["request_schema"]["properties"]

    endpoints = [
        endpoint
        for endpoint in data["endpoints"]
        if endpoint["path"] == "/v1/llmrouter/cache/stats"
    ]
    assert len(endpoints) == 1
    endpoint = endpoints[0]
    stream_block = endpoint["response_schema"].get("stream")
    assert stream_block is not None
    assert stream_block == {
        "stream_lookup_hit_total": "int",
        "stream_lookup_miss_total": "int",
        "stream_purged_total": "int",
        "stream_replay_bytes_served_total": "int",
        "stream_replay_error_total": "int",
        "stream_aborts_total": "int",
        "stream_replay_aborts_total": "int",
        "stream_live_aborts_total": "int",
        "stream_probe_tokens_spent_estimated": "int",
        "stream_replays_total": "int",
        "stream_hits": "int",
        "stream_replays_served": "int",
        "stream_replay_tokens_saved": "int",
        "stream_probe_tokens_spent": "int",
        "stream_probe_latency_ms_p50": "float",
        "stream_stored_total": "int",
        "stream_tokens_saved_total": "int",
        "stream_probes_fail_total": "int",
        "stream_probes_ok_total": "int",
    }


def test_build_app_semantic_disabled_keeps_proxy_clean(tmp_path: Path) -> None:
    settings = Settings()
    settings.semantic_cache.enabled = False
    settings.models_file = str(tmp_path / "missing-models.yaml")

    app = build_app(settings)

    proxy: ProviderProxy | None = getattr(app.state, "proxy", None)
    assert proxy is not None
    assert getattr(proxy, "_semantic_cache", None) is None
