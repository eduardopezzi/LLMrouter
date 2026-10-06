"""Tests for the E2 streaming replay path (Dev E2-B).

Covers:

* :class:`llmrouter.providers.base.BaseProvider.first_tokens` — abstract default.
* :class:`llmrouter.providers.openai_compatible.OpenAICompatibleProvider.first_tokens` —
  concrete non-streaming probe with ``max_tokens=k``.
* Runtime wiring in :func:`llmrouter.runtime._build_semantic_cache` — pass
  ``stream_*`` knobs through to :class:`SemanticCache`.
* ``GET /v1/llmrouter/cache/stats`` — exposes the ``stream`` counter block.
* ``_stream_response`` replay branch — k-token probe + replay-or-live.
* Soft-circuit on consecutive probe failures (per model).

All tests follow the TDD mandate for sprint E2: RED → GREEN. The fixture
infrastructure (semantic cache ``stream_*`` methods) is faked via duck-typed
objects because E2-A's storage branch is developed in parallel — no
hard-dependency on the schema or :class:`SemanticCache` constructor
changes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from llmrouter.api.routes import (
    _live_stream_usage,
    _maybe_replay_stream,
    _normalize_stream_chunk,
    _stream_response,
    _to_chat_request,
    create_app,
)
from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.semantic_cache import SemanticCache, StreamCacheMatch
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    Tier,
    Usage,
)
from llmrouter.providers.base import BaseProvider, ProviderError
from llmrouter.providers.openai_compatible import OpenAICompatibleProvider
from llmrouter.runtime import _build_semantic_cache

# ---------------------------------------------------------------------------
# Fakes & helpers
# ---------------------------------------------------------------------------


class _TestProvider(OpenAICompatibleProvider):
    """Concrete OpenAI-compatible provider for first_tokens tests."""

    def __init__(self, base_url: str = "http://test.invalid/v1", **kwargs: Any) -> None:
        super().__init__(name="test", api_key="key", base_url=base_url, **kwargs)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key",
        }


class _StubProviderNotImplemented(OpenAICompatibleProvider):
    """Provider whose ``first_tokens`` keeps the abstract default behaviour."""

    def __init__(self) -> None:
        super().__init__(name="stub", api_key="k", base_url="http://stub.invalid/v1")

    def _build_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": "Bearer k"}


def _chat_request(text: str = "say hi") -> ChatRequest:
    return ChatRequest(
        model="cheap",
        messages=[ChatMessage(role="user", content=text)],
        stream=True,
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
    )


def _registry() -> ModelRegistry:
    return ModelRegistry(models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),))


class _FakeEmbedder:
    """Synchronous embedder mapping texts to small deterministic vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 8
            for index, char in enumerate(text):
                vector[index % 8] += (ord(char) % 17) / 17.0
            norm = sum(value * value for value in vector) or 1.0
            vector = [value / norm**0.5 for value in vector]
            vectors.append(vector)
        return vectors


class _ScorerStub:
    """Duck-typed HybridScorer stand-in exposing ._semantic_scorer.embedder."""

    def __init__(self, embedder: object) -> None:
        self._semantic_scorer = type("_S", (), {"embedder": embedder})()


class _FakeSemanticCache:
    """Fake semantic cache exposing the stream APIs E2-A will provide.

    The test suite is written against this duck-typed surface so it can run
    independently of E2-A's storage branch.
    """

    def __init__(
        self,
        *,
        cached_chunks: list[dict[str, Any]] | None = None,
        first_k_tokens: str = "",
        completion_tokens: int = 0,
        embedder: Any | None = None,
    ) -> None:
        self._cached_chunks = cached_chunks
        self._first_k_tokens = first_k_tokens
        self._completion_tokens = completion_tokens
        self._embedder = embedder
        # Stream stats counters (E2-A naming).
        self.stream_stats_data: dict[str, int] = {
            "stream_stored_total": 0,
            "stream_purged_total": 0,
            "stream_lookup_hit_total": 0,
            "stream_lookup_miss_total": 0,
            "stream_probes_ok_total": 0,
            "stream_probes_fail_total": 0,
            "stream_replays_total": 0,
            "stream_replay_bytes_served_total": 0,
            "stream_tokens_saved_total": 0,
            "stream_replay_error_total": 0,
            "stream_aborts_total": 0,
            "stream_replay_aborts_total": 0,
            "stream_live_aborts_total": 0,
            "stream_probe_tokens_spent_estimated": 0,
        }
        self._probe_latencies: list[float] = []
        # Stream knobs (E2-A naming).
        self.stream_cache_enabled = True
        self.stream_probe_k = 8
        self.stream_probe_timeout_seconds = 10.0
        self.stream_probe_soft_circuit_threshold = 3
        self.stream_probe_soft_circuit_seconds = 3600.0
        # Behaviour flags for tests.
        self._store_calls: list[dict[str, Any]] = []
        self.lookup_calls: list[dict[str, Any]] = []
        self.hit_log_calls: list[StreamCacheMatch] = []

    async def lookup_stream_response(
        self, *args: Any, **kwargs: Any
    ) -> StreamCacheMatch | None:
        self.lookup_calls.append({"args": args, "kwargs": kwargs})
        self.stream_stats_data["stream_lookup_hit_total"] += (
            1 if self._cached_chunks is not None else 0
        )
        if self.stream_stats_data["stream_lookup_hit_total"] == 0 and self._cached_chunks is None:
            self.stream_stats_data["stream_lookup_miss_total"] += 1
            return None
        if self._cached_chunks is None:
            self.stream_stats_data["stream_lookup_miss_total"] += 1
            return None
        prompt = str(args[0]) if args else "say hi"
        return StreamCacheMatch(
            chunks=self._cached_chunks,
            prompt_tokens=5,
            completion_tokens=self._completion_tokens,
            total_tokens=5 + self._completion_tokens,
            usage_source="provider",
            first_k_tokens=self._first_k_tokens,
            prompt_text=prompt,
            prompt_hash="test-prompt-hash",
            model=str(kwargs.get("model", "cheap")),
            tier=int(kwargs.get("tier", 1)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=float(kwargs.get("top_p", 1.0)),
            max_tokens=kwargs.get("max_tokens"),
            similarity=1.0,
            threshold=0.95,
            cache_key="test-cache-key",
        )

    async def store_stream_response(self, *args: Any, **kwargs: Any) -> bool:
        self._store_calls.append({"args": args, "kwargs": kwargs})
        self.stream_stats_data["stream_stored_total"] += 1
        return True

    def stream_stats(self) -> dict[str, int | float]:
        stats: dict[str, int | float] = dict(self.stream_stats_data)
        stats.update(
            {
                "stream_hits": stats["stream_lookup_hit_total"],
                "stream_replays_served": stats["stream_replays_total"],
                "stream_replay_tokens_saved": stats["stream_tokens_saved_total"],
                "stream_probe_tokens_spent": stats[
                    "stream_probe_tokens_spent_estimated"
                ],
                "stream_probe_latency_ms_p50": (
                    sorted(self._probe_latencies)[len(self._probe_latencies) // 2]
                    if self._probe_latencies
                    else 0.0
                ),
            }
        )
        return stats

    def record_stream_probe(self, *, token_budget: int, latency_ms: float) -> None:
        self.stream_stats_data["stream_probe_tokens_spent_estimated"] += token_budget
        self._probe_latencies.append(latency_ms)

    async def record_stream_hit(self, match: StreamCacheMatch) -> bool:
        self.hit_log_calls.append(match)
        return True

    def bump_stream_counter(self, name: str, by: int = 1) -> None:
        if name in self.stream_stats_data:
            self.stream_stats_data[name] += by


class _FakeStreamingProxy:
    """Proxy stub emitting deterministic OpenAI-shape streaming chunks."""

    def __init__(self, live_chunks: list[dict[str, Any]] | None = None) -> None:
        self.last_request: ChatRequest | None = None
        self.last_decision: Any = None
        self.live_chunks: list[dict[str, Any]] = live_chunks if live_chunks is not None else [
            {
                "id": "chatcmpl-live",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "cheap",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello "},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-live",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "cheap",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "world"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]
        self.calls = 0

    async def stream_chat_completion(
        self, request: ChatRequest, decision: Any
    ) -> AsyncIterator[dict[str, Any]]:
        self.last_request = request
        self.last_decision = decision
        self.calls += 1
        for chunk in self.live_chunks:
            yield chunk


class _FakeRoutingDecision:
    """Minimal stand-in for ``MultiModelRouter.route`` output."""

    def __init__(self, primary: ModelInfo) -> None:
        self.primary = primary
        self.fallbacks: list[ModelInfo] = []
        self.tier = Tier.T1
        self.score = 1.0
        self.reason = "test"


class _FakeRouter:
    """Router stub returning a fixed decision; ``route`` is async."""

    def __init__(self, primary: ModelInfo) -> None:
        self._primary = primary
        self.routing_strategy = type(
            "_RS",
            (),
            {"value": "cost"},
        )()

    async def route(self, chat_request: ChatRequest, constraints: Any) -> _FakeRoutingDecision:
        return _FakeRoutingDecision(self._primary)


# ---------------------------------------------------------------------------
# 1) BaseProvider.first_tokens — abstract default
# ---------------------------------------------------------------------------


def test_provider_first_tokens_abstract_raises_not_implemented() -> None:
    """A provider without ``first_tokens`` impl raises ``NotImplementedError``.

    The BaseProvider default signals "I cannot probe this provider" so the
    caller can fall back to the live stream path without committing to a
    replay.
    """

    # ``_BareProvider`` extends BaseProvider directly so it truly inherits the
    # abstract default.  Extending ``_TestProvider`` (which already inherits a
    # concrete ``first_tokens`` from OpenAICompatibleProvider) would silently
    # call the concrete impl instead of the abstract default.
    class _BareProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__(name="bare", api_key="k", base_url="http://test.invalid/v1")

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

        async def chat_completion(self, request, model):
            raise NotImplementedError

        async def stream_completion(self, request, model):
            raise NotImplementedError
            yield {}  # pragma: no cover

    provider = _BareProvider()
    # Ensure the abstract default raises NotImplementedError, not silently
    # returning empty or live data.
    with pytest.raises(NotImplementedError):
        asyncio.run(provider.first_tokens(_chat_request(), "cheap", k=8))


# ---------------------------------------------------------------------------
# 2) OpenAICompatibleProvider.first_tokens — concrete impl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_first_tokens_returns_text_only_with_k_tokens() -> None:
    """Concrete first_tokens POSTs non-streaming with max_tokens=k and parses content."""
    provider = _TestProvider()

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        # Validate the request shape: non-streaming, max_tokens=k.
        json_payload = kwargs.get("json", {})
        assert json_payload["stream"] is False
        assert json_payload["max_tokens"] == 8
        return httpx.Response(
            status_code=200,
            request=httpx.Request("POST", "http://test.invalid/v1/chat/completions"),
            json={
                "id": "probe",
                "created": 1700000000,
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "Hello wo"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
            },
        )

    provider._client = httpx.AsyncClient()
    provider._client.post = mock_post  # type: ignore[assignment,method-assign]

    text = await provider.first_tokens(_chat_request(), "cheap", k=8)
    assert text == "Hello wo"
    await provider.close()


# ---------------------------------------------------------------------------
# 3) Runtime wiring — pass stream knobs into SemanticCache(...)
# ---------------------------------------------------------------------------


def test_runtime_passes_stream_knobs_to_semantic_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_build_semantic_cache`` forwards ``stream_*`` knobs to ``SemanticCache``."""
    settings = Settings()
    settings.semantic_cache.enabled = True
    settings.semantic_cache.db_path = str(tmp_path / "semantic.db")

    # Inject the stream knobs (E2-A will provide them on the config schema).
    cache_config = settings.semantic_cache
    cache_config.stream_cache_enabled = True  # type: ignore[attr-defined]
    cache_config.stream_ttl_seconds = 180.0
    cache_config.stream_probe_k = 12  # type: ignore[attr-defined]
    cache_config.stream_probe_timeout_seconds = 7.5  # type: ignore[attr-defined]
    cache_config.stream_probe_soft_circuit_threshold = 5  # type: ignore[attr-defined]
    cache_config.stream_probe_soft_circuit_seconds = 1800.0  # type: ignore[attr-defined]

    captured: dict[str, Any] = {}

    real_init = SemanticCache.__init__

    def spy_init(self: SemanticCache, *args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(SemanticCache, "__init__", spy_init)

    cache = _build_semantic_cache(settings, scorer=_ScorerStub(_FakeEmbedder()))
    assert cache is not None
    assert captured.get("stream_cache_enabled") is True
    assert captured.get("stream_ttl_seconds") == 180.0
    assert captured.get("stream_probe_k") == 12
    assert captured.get("stream_probe_timeout_seconds") == 7.5
    assert captured.get("stream_probe_soft_circuit_threshold") == 5
    assert captured.get("stream_probe_soft_circuit_seconds") == 1800.0


# ---------------------------------------------------------------------------
# 4) /v1/llmrouter/cache/stats exposes the stream counter block
# ---------------------------------------------------------------------------


def test_cache_stats_includes_stream_counters() -> None:
    """``GET /v1/llmrouter/cache/stats`` includes ``payload['stream']``."""
    cache = _FakeSemanticCache()
    app = create_app(registry=_registry(), semantic_cache=cache, api_key="secret")
    client = TestClient(app)

    response = client.get(
        "/v1/llmrouter/cache/stats",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "stream" in body
    stream = body["stream"]
    for key in (
        "stream_stored_total",
        "stream_purged_total",
        "stream_lookup_hit_total",
        "stream_lookup_miss_total",
        "stream_probes_ok_total",
        "stream_probes_fail_total",
        "stream_replays_total",
        "stream_replay_bytes_served_total",
        "stream_tokens_saved_total",
        "stream_replay_error_total",
        "stream_aborts_total",
        "stream_replay_aborts_total",
        "stream_live_aborts_total",
        "stream_probe_tokens_spent_estimated",
        "stream_hits",
        "stream_replays_served",
        "stream_replay_tokens_saved",
        "stream_probe_tokens_spent",
        "stream_probe_latency_ms_p50",
    ):
        assert key in stream
        assert stream[key] == 0


# ---------------------------------------------------------------------------
# Helpers for replay tests — wire the real _stream_response with a fake proxy
# ---------------------------------------------------------------------------


def _make_stream_request(
    *,
    semantic_cache: Any | None = None,
    selected_provider: Any | None = None,
) -> Any:
    """Build a minimal Starlette ``Request`` object for direct calls."""

    from starlette.applications import Starlette
    from starlette.datastructures import State
    from starlette.requests import Request as StarletteRequest

    app = Starlette()
    app.state = State()
    if semantic_cache is not None:
        app.state.semantic_cache = semantic_cache
    if selected_provider is not None:
        app.state.selected_provider = selected_provider
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"authorization", b"Bearer secret")],
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "http_version": "1.1",
        "scheme": "http",
        "app": app,
    }
    return StarletteRequest(scope=scope)


def _payload(*, include_usage: bool = False) -> Any:
    from llmrouter.api.routes import ChatCompletionPayload

    return ChatCompletionPayload(
        messages=[{"role": "user", "content": "say hi"}],
        model="auto",
        stream=True,
        stream_options={"include_usage": True} if include_usage else None,
    )


def test_usage_only_provider_chunk_is_preserved_and_stream_options_forwarded() -> None:
    normalized = _normalize_stream_chunk(
        {"id": "completion-1", "choices": [], "usage": {"total_tokens": 7}},
        "cheap",
    )
    assert normalized is not None
    assert normalized["choices"] == []
    assert normalized["usage"] == {"total_tokens": 7}

    chat_request = _to_chat_request(_payload(include_usage=True))
    assert chat_request.extra["stream_options"] == {"include_usage": True}
    assert "stream_options" not in _to_chat_request(_payload()).extra
    assert _normalize_stream_chunk({"choices": []}, "cheap") is None
    assert _normalize_stream_chunk({"choices": "invalid"}, "cheap") is None


def test_invalid_provider_usage_falls_back_to_estimate() -> None:
    usage, source = _live_stream_usage(
        {"prompt_tokens": "bad", "completion_tokens": 3, "total_tokens": 4},
        prompt="a prompt",
        response="some output",
    )
    assert source == "estimated"
    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 2
    assert usage.total_tokens == 4
    negative_usage, negative_source = _live_stream_usage(
        {"prompt_tokens": -1, "completion_tokens": 3, "total_tokens": 2},
        prompt="a prompt",
        response="some output",
    )
    assert negative_source == "estimated"
    assert negative_usage.prompt_tokens == 2


# ---------------------------------------------------------------------------
# (i) Replay path: byte-by-byte equivalence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_response_replays_cached_chunks_byte_by_byte() -> None:
    cached_chunks = [
        {
            "id": "chatcmpl-cached",
            "object": "chat.completion.chunk",
            "created": 1700000001,
            "model": "cheap",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "cached "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-cached",
            "object": "chat.completion.chunk",
            "created": 1700000001,
            "model": "cheap",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="cached ",
        completion_tokens=8,
    )
    # Pre-seed a probe result so the replay path proceeds.
    fake_cache.stream_stats_data["stream_lookup_hit_total"] = 1

    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    fake_proxy = _FakeStreamingProxy()

    # Mock provider whose first_tokens returns the same prefix.
    class _ReplayProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            return "cached "

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    provider = _ReplayProbeProvider()
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=fake_proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
    )
    # Attach the fake cache + provider so the replay path can find them.
    response._fake_cache = fake_cache  # type: ignore[attr-defined]
    response._fake_provider = provider  # type: ignore[attr-defined]

    # The replay generator should yield bytes equivalent to the cached chunks.
    expected_lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in cached_chunks]
    expected_lines.append("data: [DONE]\n\n")

    # Stream the body iterator.
    body_chunks: list[bytes] = []
    async for piece in response.body_iterator:
        body_chunks.append(piece.encode() if isinstance(piece, str) else piece)
    body = b"".join(body_chunks)
    assert body == b"".join(s.encode() for s in expected_lines)

    # The header ``X-LLMrouter-Cache-Status`` must be set to ``semantic_hit``.
    assert response.headers.get("x-llmrouter-cache-status") == "semantic_hit"
    assert len(fake_cache.hit_log_calls) == 1


@pytest.mark.asyncio
async def test_replay_usage_chunk_precedes_done_when_requested() -> None:
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "cached"}, "finish_reason": "stop"}]}
    ]
    cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="cached",
        completion_tokens=8,
    )
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "cached"

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache, selected_provider=_MatchingProvider()),
        chat_request=_chat_request(),
        payload=_payload(include_usage=True),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(model),  # type: ignore[arg-type]
        collector=None,
    )
    events = [
        piece.decode() if isinstance(piece, bytes) else piece
        async for piece in response.body_iterator
    ]
    assert len(events) == 3
    assert '"choices": []' in events[1]
    usage = json.loads(events[1].removeprefix("data: ").strip())
    assert usage["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 8,
        "total_tokens": 13,
    }
    assert usage["cache_status"] == "semantic_hit"
    assert usage["usage_source"] == "cached"
    assert events[2] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# (j) Probe divergence → live path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_response_falls_back_to_live_when_probe_diverges() -> None:
    cached_chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "old"},
                    "finish_reason": "stop",
                }
            ]
        }
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="OLD-PREFIX",
        completion_tokens=1,
    )
    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    fake_proxy = _FakeStreamingProxy()

    class _DivergingProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            return "new-prefix-different"

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    provider = _DivergingProbeProvider()
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=fake_proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
    )

    # The proxy MUST have been called (live path) because probe diverged.
    async for _ in response.body_iterator:
        pass
    assert fake_proxy.calls == 1
    assert fake_cache.stream_stats_data["stream_probes_fail_total"] == 1
    # No header set when falling back to live.
    assert response.headers.get("x-llmrouter-cache-status") is None
    # Replay counters must NOT increment on divergence.
    assert fake_cache.stream_stats_data["stream_replays_total"] == 0


# ---------------------------------------------------------------------------
# (k) Consumer abort during replay → store NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_response_skips_storage_when_consumer_aborts_during_replay() -> None:
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "c"}, "finish_reason": "stop"}]},
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="a",
        completion_tokens=3,
    )
    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    fake_proxy = _FakeStreamingProxy()

    class _OkProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            return "a"

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    provider = _OkProbeProvider()
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=fake_proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
    )

    # Read only the FIRST chunk then break — simulating client abort.
    iterator = response.body_iterator
    try:
        first = await iterator.__anext__()
    finally:
        await iterator.aclose()
    assert first

    # Store must NOT have been called because the consumer aborted mid-stream.
    assert fake_cache._store_calls == []
    assert fake_proxy.calls == 0
    assert fake_cache.hit_log_calls == []


@pytest.mark.asyncio
async def test_replay_metrics_wait_until_generator_resumes_after_done() -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}
    ]
    cache = _FakeSemanticCache(
        cached_chunks=chunks,
        first_k_tokens="answer",
        completion_tokens=9,
    )

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "answer"

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache, selected_provider=_MatchingProvider()),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
        ),  # type: ignore[arg-type]
        collector=None,
    )
    iterator = response.body_iterator
    assert "answer" in str(await iterator.__anext__())
    assert await iterator.__anext__() == "data: [DONE]\n\n"
    await iterator.aclose()

    assert cache.stream_stats_data["stream_replays_total"] == 0
    assert cache.stream_stats_data["stream_tokens_saved_total"] == 0
    assert cache.stream_stats_data["stream_replay_aborts_total"] == 1
    assert cache.stream_stats_data["stream_live_aborts_total"] == 0
    assert cache.hit_log_calls == []


@pytest.mark.asyncio
async def test_replay_exception_increments_legacy_error_counter() -> None:
    cache = _FakeSemanticCache(cached_chunks=[None], first_k_tokens="x")  # type: ignore[list-item]
    error_metric_calls: list[str] = []

    def fail_error_metric(name: str, by: int = 1) -> None:
        if name == "stream_replay_error_total":
            error_metric_calls.append(name)
            raise RuntimeError("metrics unavailable")
        original_bump(name, by)

    original_bump = cache.bump_stream_counter
    cache.bump_stream_counter = fail_error_metric  # type: ignore[method-assign]

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "x"

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache, selected_provider=_MatchingProvider()),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
        ),  # type: ignore[arg-type]
        collector=None,
    )

    with pytest.raises(AttributeError):
        async for _ in response.body_iterator:
            pass
    assert error_metric_calls == ["stream_replay_error_total"]


@pytest.mark.asyncio
async def test_replay_audit_and_metric_failures_do_not_break_completed_stream() -> None:
    cache = _FakeSemanticCache(
        cached_chunks=[
            {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}
        ],
        first_k_tokens="x",
    )

    async def fail_audit(match: StreamCacheMatch) -> bool:
        raise RuntimeError("audit unavailable")

    def fail_metrics(name: str, by: int = 1) -> None:
        if name == "stream_replays_total":
            raise RuntimeError("metrics unavailable")
        original_bump(name, by)

    original_bump = cache.bump_stream_counter
    cache.record_stream_hit = fail_audit  # type: ignore[method-assign]
    cache.bump_stream_counter = fail_metrics  # type: ignore[method-assign]

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "x"

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache, selected_provider=_MatchingProvider()),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
        ),  # type: ignore[arg-type]
        collector=None,
    )

    events = [piece async for piece in response.body_iterator]
    assert events[-1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# (l) X-LLMrouter-Cache-Status: semantic_hit header on replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_response_streams_with_header_cache_status_semantic_hit() -> None:
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="ok",
        completion_tokens=1,
    )
    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    fake_proxy = _FakeStreamingProxy()

    class _OkProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            return "ok"

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    provider = _OkProbeProvider()
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=fake_proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
    )

    async for _ in response.body_iterator:
        pass

    assert response.headers.get("x-llmrouter-cache-status") == "semantic_hit"


# ---------------------------------------------------------------------------
# (n) Provider without first_tokens → live path, fail counter increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_without_first_tokens_falls_back_to_live() -> None:
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="x",
        completion_tokens=1,
    )
    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)

    # ``_StubProviderNotImplemented`` keeps the abstract default → raises
    # ``NotImplementedError`` when first_tokens() is called.
    provider = _StubProviderNotImplemented()

    # We must inject the provider into the proxy so that the replay path
    # can find a stub matching ``primary.provider``.  We monkey-patch
    # ``ProviderProxy.stream_chat_completion`` via a wrapper.

    class _ProviderAwareProxy(_FakeStreamingProxy):
        def __init__(self) -> None:
            super().__init__()
            self.provider_for_probe = provider

        async def stream_chat_completion(
            self, request: ChatRequest, decision: Any
        ) -> AsyncIterator[dict[str, Any]]:
            self.last_request = request
            self.last_decision = decision
            self.calls += 1
            for chunk in self.live_chunks:
                yield chunk

    proxy = _ProviderAwareProxy()

    # The replay path needs a way to obtain the provider instance per model.
    # For these unit tests we cheat via a closure on the fake_cache instance.
    fake_cache.provider = provider  # type: ignore[attr-defined]

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
    )

    async for _ in response.body_iterator:
        pass

    # Falls back to live; fail counter incremented.
    assert proxy.calls == 1
    assert fake_cache.stream_stats_data["stream_probes_fail_total"] == 1
    assert fake_cache.stream_stats_data["stream_replays_total"] == 0


# ---------------------------------------------------------------------------
# (o) Soft-circuit: 3 consecutive probe failures → 4th skips probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_soft_circuit_after_consecutive_failures() -> None:
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="x",
        completion_tokens=1,
    )
    # Force a low threshold so we trip the circuit after 2 failures.
    fake_cache.stream_probe_soft_circuit_threshold = 2
    fake_cache.stream_probe_soft_circuit_seconds = 3600.0

    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)

    class _AlwaysFailProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            raise ProviderError("probe failed", status_code=500, provider="openai")

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    provider = _AlwaysFailProvider()
    fake_cache.provider = provider  # type: ignore[attr-defined]

    class _LocalProxy(_FakeStreamingProxy):
        pass

    proxy = _LocalProxy()

    # Soft-circuit state lives on a closure-bound dict inside
    # ``_stream_response``; for tests we share it via ``app.state``.  The
    # implementation accepts an explicit ``probe_soft_circuit`` parameter so
    # we simply thread the same dict through every request.
    shared_circuit: dict[str, list[float]] = {}

    for _ in range(2):
        response_fail = await _stream_response(
            request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
            chat_request=_chat_request(),
            payload=_payload(),
            proxy=_LocalProxy(),  # type: ignore[arg-type]
            app_router=_FakeRouter(primary),  # type: ignore[arg-type]
            collector=None,
            probe_soft_circuit=shared_circuit,
        )
        async for _ in response_fail.body_iterator:
            pass

    # Two probe failures have been recorded.
    assert fake_cache.stream_stats_data["stream_probes_fail_total"] == 2

    # Third request: circuit should now be open, so the probe must NOT run.
    probe_calls: list[int] = []

    class _CountingProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            probe_calls.append(1)
            return "x"

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    response = await _stream_response(
        request=_make_stream_request(
            semantic_cache=fake_cache, selected_provider=_CountingProbeProvider()
        ),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit=shared_circuit,
    )
    async for _ in response.body_iterator:
        pass
    assert proxy.calls == 1
    assert probe_calls == []
    assert fake_cache.stream_stats_data["stream_probes_fail_total"] == 2


@pytest.mark.asyncio
async def test_open_circuit_skips_real_semantic_cache_lookup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N2a: an open circuit bypasses the actual SemanticCache lookup method."""
    cache = SemanticCache(
        str(tmp_path / "real_open_circuit.db"),
        embedder=_FakeEmbedder(),
        stream_probe_soft_circuit_threshold=1,
    )
    lookup_calls: list[int] = []
    real_lookup = cache.lookup_stream_response

    async def spy_lookup(*args: Any, **kwargs: Any) -> Any:
        lookup_calls.append(1)
        return await real_lookup(*args, **kwargs)

    monkeypatch.setattr(cache, "lookup_stream_response", spy_lookup)
    probe_calls: list[int] = []

    class _ProbeProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            probe_calls.append(1)
            return "unused"

    result = await _maybe_replay_stream(
        semantic_cache=cache,
        selected_provider=_ProbeProvider(),
        chat_request=_chat_request(),
        selected_model=ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )

    assert result is None
    assert lookup_calls == []
    assert probe_calls == []


@pytest.mark.asyncio
async def test_legacy_tuple_lookup_remains_supported_when_probe_metrics_fail() -> None:
    cache = _FakeSemanticCache()
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}
    ]

    async def legacy_lookup(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], int, str]:
        return cached_chunks, 3, "x"

    def fail_probe_metrics(*, token_budget: int, latency_ms: float) -> None:
        raise RuntimeError("metrics unavailable")

    cache.lookup_stream_response = legacy_lookup  # type: ignore[method-assign]
    cache.record_stream_probe = fail_probe_metrics  # type: ignore[method-assign]

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "x"

    result = await _maybe_replay_stream(
        semantic_cache=cache,
        selected_provider=_MatchingProvider(),
        chat_request=_chat_request(),
        selected_model=ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),
        probe_soft_circuit={},
    )
    assert result is not None
    assert result.chunks == cached_chunks
    assert result.usage.completion_tokens == 3
    assert result.usage_source == "estimated"
    assert result.audit_match is None

    cache.record_stream_probe = None  # type: ignore[method-assign]
    result_without_metrics = await _maybe_replay_stream(
        semantic_cache=cache,
        selected_provider=_MatchingProvider(),
        chat_request=_chat_request(),
        selected_model=ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),
        probe_soft_circuit={},
    )
    assert result_without_metrics is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_tuple, expose_audit", [(True, True), (False, False)])
async def test_replay_supports_legacy_cache_and_optional_audit_method(
    legacy_tuple: bool, expose_audit: bool
) -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "cached"}, "finish_reason": "stop"}]}
    ]
    cache = _FakeSemanticCache(cached_chunks=chunks, first_k_tokens="cached")
    if not expose_audit:
        cache.record_stream_hit = None  # type: ignore[method-assign]
    if legacy_tuple:
        async def legacy_lookup(
            *args: Any, **kwargs: Any
        ) -> tuple[list[dict[str, Any]], int, str]:
            return chunks, 5, "cached"

        cache.lookup_stream_response = legacy_lookup  # type: ignore[method-assign]

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "cached"

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache, selected_provider=_MatchingProvider()),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
        ),  # type: ignore[arg-type]
        collector=None,
    )
    events = [piece async for piece in response.body_iterator]
    assert events[-1] == "data: [DONE]\n\n"
    assert cache.stream_stats_data["stream_replays_total"] == 1

# ---------------------------------------------------------------------------
# (p) Stream stats counters after replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_stats_increment_properly_after_replay() -> None:
    cached_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "ab"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "cd"}, "finish_reason": "stop"}]},
    ]
    fake_cache = _FakeSemanticCache(
        cached_chunks=cached_chunks,
        first_k_tokens="ab",
        completion_tokens=4,
    )
    primary = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    fake_proxy = _FakeStreamingProxy()

    class _OkProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            return "ab"

        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json", "Authorization": "Bearer k"}

    provider = _OkProbeProvider()
    fake_cache.provider = provider  # type: ignore[attr-defined]

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=fake_cache, selected_provider=provider),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=fake_proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(primary),  # type: ignore[arg-type]
        collector=None,
    )
    async for _ in response.body_iterator:
        pass

    stats = fake_cache.stream_stats()
    assert stats["stream_replays_total"] == 1
    assert stats["stream_probes_ok_total"] == 1
    assert stats["stream_replay_bytes_served_total"] > 0
    assert stats["stream_tokens_saved_total"] == 4


@pytest.mark.asyncio
async def test_live_stream_emits_provider_usage_before_done_and_caches_cleanly() -> None:
    usage = {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    proxy = _FakeStreamingProxy(
        live_chunks=[
            {
                "choices": [
                    {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ]
            },
            {"choices": [], "usage": usage},
        ]
    )
    cache = _FakeSemanticCache()
    cache.stream_probe_soft_circuit_threshold = 1
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache),
        chat_request=_chat_request(),
        payload=_payload(include_usage=True),
        proxy=proxy,  # type: ignore[arg-type]
        app_router=_FakeRouter(model),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )
    events = [
        piece.decode() if isinstance(piece, bytes) else piece
        async for piece in response.body_iterator
    ]
    assert len(events) == 4
    usage_event = json.loads(events[-2].removeprefix("data: ").strip())
    assert usage_event["usage"] == usage
    assert usage_event["cache_status"] == "live"
    assert usage_event["usage_source"] == "provider"
    assert events[-1] == "data: [DONE]\n\n"
    assert cache._store_calls[0]["kwargs"]["usage_source"] == "provider"


@pytest.mark.asyncio
async def test_incomplete_live_stream_is_never_stored() -> None:
    cache = _FakeSemanticCache()
    cache.stream_probe_soft_circuit_threshold = 1
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)

    class _InterruptedProxy(_FakeStreamingProxy):
        async def stream_chat_completion(
            self, request: ChatRequest, decision: Any
        ) -> AsyncIterator[dict[str, Any]]:
            self.calls += 1
            yield {
                "choices": [
                    {"index": 0, "delta": {"content": "partial"}, "finish_reason": None}
                ]
            }
            raise ProviderError("provider disconnected", status_code=502, provider="test")

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_InterruptedProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(model),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )
    events = [piece async for piece in response.body_iterator]
    assert any('"type": "provider_error"' in str(event) for event in events)
    assert cache._store_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ]
            }
        ],
    ],
    ids=["empty", "terminal-without-content"],
)
async def test_empty_or_contentless_live_stream_is_never_stored(
    chunks: list[dict[str, Any]],
) -> None:
    cache = _FakeSemanticCache()
    cache.stream_probe_soft_circuit_threshold = 1
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(live_chunks=chunks),  # type: ignore[arg-type]
        app_router=_FakeRouter(model),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )

    events = [piece async for piece in response.body_iterator]
    assert events[-1] == "data: [DONE]\n\n"
    assert cache._store_calls == []


@pytest.mark.asyncio
async def test_live_stream_exception_after_terminal_does_not_store() -> None:
    cache = _FakeSemanticCache()
    cache.stream_probe_soft_circuit_threshold = 1
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)

    class _BrokenAfterTerminalProxy(_FakeStreamingProxy):
        async def stream_chat_completion(
            self, request: ChatRequest, decision: Any
        ) -> AsyncIterator[dict[str, Any]]:
            self.calls += 1
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "partial"},
                        "finish_reason": "stop",
                    }
                ]
            }
            raise RuntimeError("provider iterator failed after terminal chunk")

    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_BrokenAfterTerminalProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(model),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )

    iterator = response.body_iterator
    assert await iterator.__anext__()
    with pytest.raises(RuntimeError, match="after terminal chunk"):
        await iterator.__anext__()
    assert cache._store_calls == []


@pytest.mark.asyncio
async def test_generator_exit_on_live_stream_counts_abort_without_storing() -> None:
    cache = _FakeSemanticCache()
    cache.stream_probe_soft_circuit_threshold = 1
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(model),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )
    iterator = response.body_iterator
    assert await iterator.__anext__()
    await iterator.aclose()
    assert cache._store_calls == []
    assert cache.stream_stats_data["stream_aborts_total"] == 1
    assert cache.stream_stats_data["stream_live_aborts_total"] == 1
    assert cache.stream_stats_data["stream_replay_aborts_total"] == 0
    assert cache.stream_stats_data["stream_replay_error_total"] == 1


@pytest.mark.asyncio
async def test_abort_metrics_are_best_effort_independently() -> None:
    cache = _FakeSemanticCache()
    cache.stream_probe_soft_circuit_threshold = 1
    original_bump = cache.bump_stream_counter

    def fail_aggregate(name: str, by: int = 1) -> None:
        if name == "stream_aborts_total":
            raise RuntimeError("aggregate metric unavailable")
        original_bump(name, by)

    cache.bump_stream_counter = fail_aggregate  # type: ignore[method-assign]
    response = await _stream_response(
        request=_make_stream_request(semantic_cache=cache),
        chat_request=_chat_request(),
        payload=_payload(),
        proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
        app_router=_FakeRouter(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
        ),  # type: ignore[arg-type]
        collector=None,
        probe_soft_circuit={"cheap": [time.monotonic()]},
    )
    iterator = response.body_iterator
    assert await iterator.__anext__()
    await iterator.aclose()

    assert cache.stream_stats_data["stream_aborts_total"] == 0
    assert cache.stream_stats_data["stream_live_aborts_total"] == 1
    assert cache.stream_stats_data["stream_replay_error_total"] == 1


@pytest.mark.asyncio
async def test_real_stream_replay_hit_log_only_after_done_and_best_effort(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "stream_replay_audit.db")
    cache = SemanticCache(
        db_path,
        embedder=_FakeEmbedder(),
        threshold=0.99,
        hit_log_enabled=True,
    )
    model = ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1)
    chunks = [
        {
            "id": "cached-id",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "cheap",
            "choices": [
                {"index": 0, "delta": {"content": "cached answer"}, "finish_reason": "stop"}
            ],
        }
    ]
    assert await cache.store_stream_response(
        chunks,
        prompt="say hi",
        usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        model="cheap",
        tier=int(Tier.T1),
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
        usage_source="provider",
    )

    class _MatchingProvider(_TestProvider):
        async def first_tokens(self, request: ChatRequest, name: str, k: int) -> str:
            return "cached answer"

    async def make_response() -> Any:
        return await _stream_response(
            request=_make_stream_request(
                semantic_cache=cache, selected_provider=_MatchingProvider()
            ),
            chat_request=_chat_request(),
            payload=_payload(),
            proxy=_FakeStreamingProxy(),  # type: ignore[arg-type]
            app_router=_FakeRouter(model),  # type: ignore[arg-type]
            collector=None,
            probe_soft_circuit={},
        )

    response = await make_response()
    events = [piece async for piece in response.body_iterator]
    assert events[-1] == "data: [DONE]\n\n"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT prompt_text, response_text, model, verified "
            "FROM semantic_cache_hit_log"
        ).fetchall()
    assert rows == [("say hi", "cached answer", "cheap", 0)]

    # A second replay stopped while suspended at its first chunk must not log.
    aborted = await make_response()
    iterator = aborted.body_iterator
    assert await iterator.__anext__()
    await iterator.aclose()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM semantic_cache_hit_log").fetchone()[0] == 1

    # Hit-log storage failure remains best-effort after a fully emitted replay.
    def fail_write(_: dict[str, Any]) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(cache, "_write_hit_log_row", fail_write)
    failed_audit = await make_response()
    failed_events = [piece async for piece in failed_audit.body_iterator]
    assert failed_events[-1] == "data: [DONE]\n\n"
    assert cache.stream_stats()["stream_replays_total"] == 2
