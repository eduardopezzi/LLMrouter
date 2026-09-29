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
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from llmrouter.api.routes import _stream_response, create_app
from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.semantic_cache import SemanticCache
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    Tier,
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
        }
        # Stream knobs (E2-A naming).
        self.stream_cache_enabled = True
        self.stream_probe_k = 8
        self.stream_probe_timeout_seconds = 10.0
        self.stream_probe_soft_circuit_threshold = 3
        self.stream_probe_soft_circuit_seconds = 3600.0
        # Behaviour flags for tests.
        self._store_calls: list[dict[str, Any]] = []
        self.lookup_calls: list[dict[str, Any]] = []

    async def lookup_stream_response(self, *args: Any, **kwargs: Any) -> tuple | None:
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
        return (self._cached_chunks, self._completion_tokens, self._first_k_tokens)

    async def store_stream_response(self, *args: Any, **kwargs: Any) -> bool:
        self._store_calls.append({"args": args, "kwargs": kwargs})
        self.stream_stats_data["stream_stored_total"] += 1
        return True

    def stream_stats(self) -> dict[str, int]:
        return dict(self.stream_stats_data)

    def bump_stream_counter(self, name: str, by: int = 1) -> None:
        if name in self.stream_stats_data:
            self.stream_stats_data[name] += by


class _FakeStreamingProxy:
    """Proxy stub emitting deterministic OpenAI-shape streaming chunks."""

    def __init__(self, live_chunks: list[dict[str, Any]] | None = None) -> None:
        self.last_request: ChatRequest | None = None
        self.last_decision: Any = None
        self.live_chunks: list[dict[str, Any]] = live_chunks or [
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


def _payload() -> Any:
    from llmrouter.api.routes import ChatCompletionPayload

    return ChatCompletionPayload(
        messages=[{"role": "user", "content": "say hi"}],
        model="auto",
        stream=True,
    )


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
    # We swap in a probe that records whether it was called via a list.
    probe_calls: list[int] = []

    class _CountingProbeProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            super().__init__(name="openai", api_key="k", base_url="http://test.invalid/v1")

        async def first_tokens(self, request: ChatRequest, model: str, k: int) -> str:
            probe_calls.append(1)
            return "x"  # would match, but the circuit should block us

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
    # The probe must NOT have been invoked while the circuit is open.
    assert probe_calls == []
    # The fail counter stays at the prior 2 (the short-circuit returns without
    # bumping it because the probe never ran).
    assert fake_cache.stream_stats_data["stream_probes_fail_total"] == 2


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
