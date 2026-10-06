"""Tests for POST /v1/llmrouter/cache/verify (P-CHR judge run).

Covers:
- 503 when the semantic cache is not configured;
- API-key enforcement (401);
- 200 payload passthrough of ``verify_pending`` with default sample size,
  judge construction from settings and judge/sample capture;
- body ``sample_size`` respected and validated (422 for <= 0);
- judge failures mapped to 500;
- the orchestrator amendment: ``verify_error`` rows (verified=2) are
  re-audited on the next run (only ok/mismatch are final);
- runtime wiring: ``_build_semantic_cache`` forwards ``hit_log_enabled``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llmrouter.api.routes import create_app
from llmrouter.config import Settings, get_settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.semantic_cache import OllamaJudge, SemanticCache
from llmrouter.core.types import ChatResponse, ModelInfo, Provider, Tier, Usage
from llmrouter.runtime import _build_semantic_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(name="summary", provider=Provider.OPENAI, tier=Tier.T1),
        )
    )


def _response(content: str = "Paris") -> ChatResponse:
    return ChatResponse(
        id="resp-1",
        model="gpt-4o",
        choices=[{"message": {"role": "assistant", "content": content}}],
        usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        latency_ms=50.0,
    )


_VERIFY_PAYLOAD: dict[str, Any] = {
    "checked": 2,
    "ok": 1,
    "mismatch": 1,
    "error": 0,
    "buckets": {
        "0.90-0.95": {"checked": 1, "ok": 0, "mismatch": 1},
        "0.95+": {"checked": 1, "ok": 1, "mismatch": 0},
    },
}


class _StubSemanticCache:
    """Duck-typed SemanticCache capturing verify_pending calls."""

    def __init__(
        self, payload: dict[str, Any] | None = None, *, error: Exception | None = None
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.payload = payload if payload is not None else _VERIFY_PAYLOAD
        self._error = error

    async def verify_pending(
        self,
        sample_size: int,
        judge: Any,
        method: str = "local_llm",
    ) -> dict[str, Any]:
        self.calls.append(
            {"sample_size": sample_size, "judge": judge, "method": method}
        )
        if self._error is not None:
            raise self._error
        return dict(self.payload)


class _FakeSyncEmbedder:
    """Synchronous embedder returning fixed 8-dim vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * 8 for _ in texts]


class _ScorerStub:
    """Duck-typed HybridScorer stand-in exposing ._semantic_scorer.embedder."""

    def __init__(self, embedder: object) -> None:
        self._semantic_scorer = type("_S", (), {"embedder": embedder})()


def _verified_flag(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(
            conn.execute("SELECT verified FROM semantic_cache_hit_log").fetchone()[0]
        )


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/llmrouter/cache/verify
# ---------------------------------------------------------------------------


class TestVerifyEndpoint:
    def test_verify_returns_503_without_semantic_cache(self) -> None:
        app = create_app(registry=_registry())
        client = TestClient(app)

        response = client.post("/v1/llmrouter/cache/verify")

        assert response.status_code == 503
        assert "Semantic cache" in response.json()["detail"]

    def test_verify_requires_api_key(self) -> None:
        cache = _StubSemanticCache()
        app = create_app(registry=_registry(), semantic_cache=cache, api_key="secret")
        client = TestClient(app)

        missing = client.post("/v1/llmrouter/cache/verify")
        assert missing.status_code == 401

        wrong = client.post(
            "/v1/llmrouter/cache/verify", headers={"X-Api-Key": "not-the-key"}
        )
        assert wrong.status_code == 401
        assert cache.calls == []

    def test_verify_returns_verify_pending_payload(self) -> None:
        cache = _StubSemanticCache()
        app = create_app(registry=_registry(), semantic_cache=cache)
        client = TestClient(app)

        response = client.post("/v1/llmrouter/cache/verify")

        assert response.status_code == 200
        assert response.json() == _VERIFY_PAYLOAD
        assert len(cache.calls) == 1
        call = cache.calls[0]
        assert call["method"] == "local_llm"
        assert call["sample_size"] == get_settings().semantic_cache.verify_sample_size
        assert isinstance(call["judge"], OllamaJudge)

    def test_verify_body_sample_size_is_respected(self) -> None:
        cache = _StubSemanticCache()
        app = create_app(registry=_registry(), semantic_cache=cache)
        client = TestClient(app)

        response = client.post("/v1/llmrouter/cache/verify", json={"sample_size": 3})

        assert response.status_code == 200
        assert cache.calls[0]["sample_size"] == 3

    def test_verify_body_sample_size_validated(self) -> None:
        cache = _StubSemanticCache()
        app = create_app(registry=_registry(), semantic_cache=cache)
        client = TestClient(app)

        for bad in (0, -2):
            response = client.post(
                "/v1/llmrouter/cache/verify", json={"sample_size": bad}
            )
            assert response.status_code == 422, f"sample_size={bad}"
        assert cache.calls == []

    def test_verify_maps_judge_failure_to_500(self) -> None:
        cache = _StubSemanticCache(error=RuntimeError("judge offline"))
        app = create_app(registry=_registry(), semantic_cache=cache)
        client = TestClient(app)

        response = client.post("/v1/llmrouter/cache/verify")

        assert response.status_code == 500
        assert "judge offline" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Orchestrator amendment: verify_error rows are re-audited
# ---------------------------------------------------------------------------


class TestVerifyPendingRetriesErrors:
    @pytest.mark.asyncio
    async def test_verify_error_rows_are_retried_next_run(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "verify_retry.db")
        cache = SemanticCache(db_path, hit_log_enabled=True)
        stored = await cache.store(
            _response(),
            "What is the capital of France?",
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
            embedding=[1.0, 0.0, 0.0],
        )
        assert stored
        hit = await cache.lookup(
            [1.0, 0.0, 0.0],
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert hit is not None

        class ExplodingJudge:
            def __call__(self, prompt: Any, response: Any) -> tuple[bool, float, str]:
                raise RuntimeError("judge offline")

        first = await cache.verify_pending(10, ExplodingJudge())
        assert first["checked"] == 1
        assert first["error"] == 1
        assert _verified_flag(db_path) == 2

        class OkJudge:
            def __call__(
                self, prompt: Any, response: Any
            ) -> tuple[bool, float | None, str | None]:
                return True, 0.9, "ok"

        # Amendment: verify_error rows are re-selected (not burned forever).
        second = await cache.verify_pending(10, OkJudge())
        assert second["checked"] == 1
        assert second["ok"] == 1
        assert _verified_flag(db_path) == 1

        # Final verdicts stay final: ok is never re-checked.
        third = await cache.verify_pending(10, OkJudge())
        assert third["checked"] == 0


# ---------------------------------------------------------------------------
# Runtime wiring: _build_semantic_cache forwards hit_log_enabled
# ---------------------------------------------------------------------------


class TestRuntimeWiring:
    def test_build_semantic_cache_forwards_hit_log_enabled(
        self, tmp_path: Path
    ) -> None:
        settings = Settings()
        settings.semantic_cache.enabled = True
        settings.semantic_cache.db_path = str(tmp_path / "semantic.db")
        settings.semantic_cache.hit_log_enabled = True

        cache = _build_semantic_cache(settings, scorer=_ScorerStub(_FakeSyncEmbedder()))

        assert cache is not None
        assert cache._hit_log_enabled is True

    def test_build_semantic_cache_hit_log_can_be_disabled(
        self, tmp_path: Path
    ) -> None:
        settings = Settings()
        settings.semantic_cache.enabled = True
        settings.semantic_cache.db_path = str(tmp_path / "semantic.db")
        settings.semantic_cache.hit_log_enabled = False

        cache = _build_semantic_cache(settings, scorer=_ScorerStub(_FakeSyncEmbedder()))

        assert cache is not None
        assert cache._hit_log_enabled is False
