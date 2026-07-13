"""Tests for ProviderProxy streaming and fallback logic."""

from __future__ import annotations

from typing import Any
from collections.abc import AsyncIterator

import pytest

from llmrouter.core.health import ModelHealthTracker
from llmrouter.core.proxy import ProviderProxy, _unique_attempts
from llmrouter.core.stats import MetricsCollector
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ModelInfo,
    Provider,
    RoutingDecision,
    Tier,
    Usage,
)
from llmrouter.providers.base import BaseProvider, ProviderError


class StubProvider(BaseProvider):
    """In-memory provider for testing — returns canned responses."""

    def __init__(
        self,
        name: str = "stub",
        *,
        response: ChatResponse | None = None,
        chunks: list[dict[str, Any]] | None = None,
        error: ProviderError | None = None,
        supports_stream: bool = True,
    ) -> None:
        super().__init__(name)
        self._response = response
        self._chunks = chunks
        self._error = error
        self._supports_stream = supports_stream

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        if self._error is not None:
            raise self._error
        if self._response is None:
            return ChatResponse(
                id="test-id",
                model=model,
                choices=[{"message": {"role": "assistant", "content": "ok"}}],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason=FinishReason.STOP,
            )
        return self._response

    async def stream_completion(
        self, request: ChatRequest, model: str
    ) -> AsyncIterator[dict[str, object]]:
        if not self._supports_stream:
            raise ProviderError("No streaming", status_code=501, provider=self._name)
        if self._error is not None:
            raise self._error
        if self._chunks is not None:
            for chunk in self._chunks:
                yield chunk
        else:
            yield {"choices": [{"delta": {"content": "hello"}}]}


def _model(name: str, provider: Provider = Provider.OPENAI, tier: Tier = Tier.T2) -> ModelInfo:
    return ModelInfo(name=name, provider=provider, tier=tier)


def _request() -> ChatRequest:
    return ChatRequest(model=None, messages=[ChatMessage(role="user", content="hi")])


def _decision(primary: ModelInfo, fallbacks: list[ModelInfo] | None = None) -> RoutingDecision:
    return RoutingDecision(
        primary=primary,
        fallbacks=fallbacks or [],
        score=0.5,
        tier=primary.tier,
        reason="test",
    )


@pytest.mark.asyncio
async def test_chat_completion_success() -> None:
    provider = StubProvider("openai")
    proxy = ProviderProxy({Provider.OPENAI: provider})
    model = _model("gpt-4o")
    response = await proxy.chat_completion(_request(), _decision(model))
    assert response.model == "gpt-4o"
    assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_chat_completion_fallback_on_error() -> None:
    primary = _model("primary", Provider.OPENAI)
    fallback = _model("fallback", Provider.OLLAMA)
    provider1 = StubProvider("openai", error=ProviderError("fail", status_code=500, provider="openai"))
    provider2 = StubProvider("ollama")
    proxy = ProviderProxy(
        {Provider.OPENAI: provider1, Provider.OLLAMA: provider2},
    )
    response = await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    assert response.model == "fallback"


@pytest.mark.asyncio
async def test_metrics_collects_completed_request_and_provider_fallback() -> None:
    primary = _model("primary", Provider.OPENAI)
    fallback = _model("fallback", Provider.OLLAMA)
    metrics = MetricsCollector()
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 500, "openai")),
            Provider.OLLAMA: StubProvider("ollama"),
        },
        metrics_collector=metrics,
    )

    response = await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    snapshot = await metrics.snapshot()

    assert response.model == "fallback"
    assert snapshot.total_requests == 1
    assert snapshot.fallback_available == 1
    assert snapshot.fallback_used == 1
    assert snapshot.failed_requests == 0
    assert snapshot.tier_distribution == {"tier_2": 1}
    assert snapshot.latency["sample_count"] == 1
    assert snapshot.errors["by_provider"] == {"openai": 1}
    assert snapshot.errors["by_model"] == {"primary": 1}


@pytest.mark.asyncio
async def test_metrics_collects_final_provider_failure() -> None:
    model = _model("primary", Provider.OPENAI)
    metrics = MetricsCollector()
    proxy = ProviderProxy(
        {Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 503, "openai"))},
        metrics_collector=metrics,
    )

    with pytest.raises(ProviderError):
        await proxy.chat_completion(_request(), _decision(model))

    snapshot = await metrics.snapshot()
    assert snapshot.total_requests == 1
    assert snapshot.failed_requests == 1
    assert snapshot.fallback_used == 0
    assert snapshot.errors["by_provider"] == {"openai": 1}


@pytest.mark.asyncio
async def test_chat_completion_all_providers_fail() -> None:
    primary = _model("primary", Provider.OPENAI)
    fallback = _model("fallback", Provider.OPENAI)
    error = ProviderError("fail", status_code=503, provider="openai")
    proxy = ProviderProxy(
        {Provider.OPENAI: StubProvider("openai", error=error)},
    )
    with pytest.raises(ProviderError) as exc_info:
        await proxy.chat_completion(
            _request(),
            _decision(primary, [fallback]),
        )
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_chat_completion_disabled_provider_skipped() -> None:
    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {Provider.OPENAI: StubProvider("openai"), Provider.OLLAMA: StubProvider("ollama")},
    )
    proxy.disable_provider(Provider.OPENAI)
    response = await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    assert response.model == "fb"


@pytest.mark.asyncio
async def test_chat_completion_provider_not_configured() -> None:
    primary = _model("p", Provider.GEMINI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {Provider.OLLAMA: StubProvider("ollama")},
    )
    response = await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    assert response.model == "fb"


@pytest.mark.asyncio
async def test_chat_completion_no_attempts_available() -> None:
    primary = _model("p", Provider.GEMINI)
    proxy = ProviderProxy({})
    with pytest.raises(ProviderError) as exc_info:
        await proxy.chat_completion(_request(), _decision(primary))
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_stream_completion_success() -> None:
    provider = StubProvider("openai", chunks=[{"choices": [{"delta": {"content": "hi"}}]}])
    proxy = ProviderProxy({Provider.OPENAI: provider})
    model = _model("gpt-4o")
    chunks = []
    async for chunk in proxy.stream_chat_completion(_request(), _decision(model)):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"


@pytest.mark.asyncio
async def test_stream_completion_fallback_on_error() -> None:
    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 500, "openai")),
            Provider.OLLAMA: StubProvider("ollama", chunks=[{"choices": [{"delta": {"content": "ok"}}]}]),
        },
    )
    chunks = []
    async for chunk in proxy.stream_chat_completion(_request(), _decision(primary, [fallback])):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "ok"


@pytest.mark.asyncio
async def test_metrics_collects_streaming_request() -> None:
    model = _model("stream", Provider.OPENAI)
    metrics = MetricsCollector()
    proxy = ProviderProxy(
        {Provider.OPENAI: StubProvider("openai")},
        metrics_collector=metrics,
    )

    async for _ in proxy.stream_chat_completion(_request(), _decision(model)):
        pass

    snapshot = await metrics.snapshot()
    assert snapshot.total_requests == 1
    assert snapshot.stream_requests == 1
    assert snapshot.stream_fallback_used == 0


@pytest.mark.asyncio
async def test_stream_completion_all_fail() -> None:
    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 503, "openai")),
            Provider.OLLAMA: StubProvider("ollama", error=ProviderError("fail", 503, "ollama")),
        },
    )
    chunks = []
    with pytest.raises(ProviderError):
        async for chunk in proxy.stream_chat_completion(_request(), _decision(primary, [fallback])):
            chunks.append(chunk)


@pytest.mark.asyncio
async def test_stream_completion_provider_not_configured() -> None:
    primary = _model("p", Provider.GEMINI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {Provider.OLLAMA: StubProvider("ollama", chunks=[{"choices": [{"delta": {"content": "ok"}}]}])},
    )
    chunks = []
    async for chunk in proxy.stream_chat_completion(_request(), _decision(primary, [fallback])):
        chunks.append(chunk)
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_stream_completion_no_attempts() -> None:
    primary = _model("p", Provider.GEMINI)
    proxy = ProviderProxy({})
    with pytest.raises(ProviderError):
        async for _ in proxy.stream_chat_completion(_request(), _decision(primary)):
            pass


@pytest.mark.asyncio
async def test_stream_completion_disabled_provider() -> None:
    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", chunks=[{"choices": [{"delta": {"content": "x"}}]}]),
            Provider.OLLAMA: StubProvider("ollama", chunks=[{"choices": [{"delta": {"content": "y"}}]}]),
        },
    )
    proxy.disable_provider(Provider.OPENAI)
    chunks = []
    async for chunk in proxy.stream_chat_completion(_request(), _decision(primary, [fallback])):
        chunks.append(chunk)
    assert chunks[0]["choices"][0]["delta"]["content"] == "y"


@pytest.mark.asyncio
async def test_close_closes_all_providers() -> None:
    provider = StubProvider("openai")
    proxy = ProviderProxy({Provider.OPENAI: provider})
    await proxy.close()
    assert provider._client is None


@pytest.mark.asyncio
async def test_provider_error_callback_invoked() -> None:
    callback_calls: list[tuple[Any, ProviderError]] = []

    def on_error(model: Any, exc: ProviderError) -> None:
        callback_calls.append((model, exc))

    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 500, "openai")),
            Provider.OLLAMA: StubProvider("ollama"),
        },
        on_provider_error=on_error,
    )
    await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    assert len(callback_calls) == 1
    assert callback_calls[0][1].status_code == 500


@pytest.mark.asyncio
async def test_provider_error_callback_exception_handled() -> None:
    def bad_callback(model: Any, exc: ProviderError) -> None:
        raise RuntimeError("callback failure")

    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 500, "openai")),
            Provider.OLLAMA: StubProvider("ollama"),
        },
        on_provider_error=bad_callback,
    )
    response = await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    assert response.model == "fb"


@pytest.mark.asyncio
async def test_health_tracker_records_success() -> None:
    tracker = ModelHealthTracker()
    provider = StubProvider("openai")
    proxy = ProviderProxy({Provider.OPENAI: provider}, health_tracker=tracker)
    model = _model("gpt-4o")
    await proxy.chat_completion(_request(), _decision(model))
    health = await tracker.get_health("gpt-4o")
    assert health.request_count == 1


@pytest.mark.asyncio
async def test_health_tracker_records_error() -> None:
    tracker = ModelHealthTracker()
    primary = _model("p", Provider.OPENAI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.OPENAI: StubProvider("openai", error=ProviderError("fail", 500, "openai")),
            Provider.OLLAMA: StubProvider("ollama"),
        },
        health_tracker=tracker,
    )
    await proxy.chat_completion(_request(), _decision(primary, [fallback]))
    health = await tracker.get_health("p")
    assert health.request_count == 1
    assert health.error_rate > 0


@pytest.mark.asyncio
async def test_health_tracker_records_stream_success() -> None:
    tracker = ModelHealthTracker()
    provider = StubProvider("openai", chunks=[{"choices": [{"delta": {"content": "hi"}}]}])
    proxy = ProviderProxy({Provider.OPENAI: provider}, health_tracker=tracker)
    model = _model("gpt-4o")
    async for _ in proxy.stream_chat_completion(_request(), _decision(model)):
        pass
    health = await tracker.get_health("gpt-4o")
    assert health.request_count == 1


@pytest.mark.asyncio
async def test_stream_completion_no_streaming_support() -> None:
    """Provider without stream_completion should trigger fallback."""
    class NoStreamProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__("custom")

        async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
            return ChatResponse(id="id", model=model, choices=[{"message": {"content": "x"}}],
                                usage=Usage(), finish_reason=FinishReason.STOP)

        async def stream_completion(self, request: ChatRequest, model: str) -> AsyncIterator[dict[str, object]]:
            yield {}

    primary = _model("p", Provider.GEMINI)
    fallback = _model("fb", Provider.OLLAMA)
    proxy = ProviderProxy(
        {
            Provider.GEMINI: NoStreamProvider(),
            Provider.OLLAMA: StubProvider("ollama", chunks=[{"choices": [{"delta": {"content": "ok"}}]}]),
        },
    )
    chunks = []
    async for chunk in proxy.stream_chat_completion(_request(), _decision(primary, [fallback])):
        chunks.append(chunk)
    assert len(chunks) >= 1


def test_unique_attempts_deduplicates() -> None:
    m1 = _model("a")
    m2 = _model("b")
    m3 = _model("a")  # duplicate of m1
    result = _unique_attempts([m1, m2, m3])
    assert len(result) == 2
    assert result[0].name == "a"
    assert result[1].name == "b"


def test_proxy_providers_property() -> None:
    proxy = ProviderProxy(
        {Provider.OPENAI: StubProvider("openai"), Provider.OLLAMA: StubProvider("ollama")},
    )
    assert Provider.OPENAI in proxy.providers
    assert Provider.OLLAMA in proxy.providers

    proxy.disable_provider(Provider.OPENAI)
    assert Provider.OPENAI not in proxy.providers
    assert Provider.OLLAMA in proxy.providers


@pytest.mark.asyncio
async def test_health_tracker_exception_handled() -> None:
    """If health tracker raises, proxy should still return response."""
    class BadTracker(ModelHealthTracker):
        async def record_success(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("tracker failure")

        async def record_error(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("tracker failure")

    tracker = BadTracker()
    provider = StubProvider("openai")
    proxy = ProviderProxy({Provider.OPENAI: provider}, health_tracker=tracker)
    model = _model("gpt-4o")
    response = await proxy.chat_completion(_request(), _decision(model))
    assert response.model == "gpt-4o"


@pytest.mark.asyncio
async def test_estimate_cost() -> None:
    model = ModelInfo(
        name="test",
        provider=Provider.OPENAI,
        tier=Tier.T2,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = ProviderProxy._estimate_cost(model, usage)
    assert cost == pytest.approx(0.01 + 0.01)  # 1000/1000 * 0.01 + 500/1000 * 0.02
