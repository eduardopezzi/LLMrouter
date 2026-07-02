from __future__ import annotations

import pytest

from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingDecision,
    Tier,
)
from llmrouter.providers.base import ProviderError


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, request: ChatRequest, model: str) -> object:
        self.calls += 1
        raise ProviderError("failed", status_code=500, provider="openai")


class SuccessfulProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, request: ChatRequest, model: str) -> object:
        self.calls += 1
        return object()


@pytest.mark.asyncio
async def test_proxy_deduplicates_repeated_model_attempts() -> None:
    provider = FailingProvider()
    proxy = ProviderProxy({Provider.OPENAI: provider})
    model = ModelInfo(name="openai/test", provider=Provider.OPENAI, tier=Tier.T3)
    decision = RoutingDecision(
        primary=model,
        fallbacks=[model],
        score=0.0,
        tier=Tier.T3,
        reason="test",
    )

    with pytest.raises(ProviderError):
        await proxy.chat_completion(
            ChatRequest(model=None, messages=[ChatMessage(role="user", content="hello")]),
            decision,
        )

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_proxy_notifies_provider_error_callback() -> None:
    provider = FailingProvider()
    model = ModelInfo(name="openai/test", provider=Provider.OPENAI, tier=Tier.T3)
    seen: list[tuple[str, int]] = []
    proxy = ProviderProxy(
        {Provider.OPENAI: provider},
        on_provider_error=lambda failed_model, exc: seen.append(
            (failed_model.name, exc.status_code)
        ),
    )
    decision = RoutingDecision(
        primary=model,
        fallbacks=[],
        score=0.0,
        tier=Tier.T3,
        reason="test",
    )

    with pytest.raises(ProviderError):
        await proxy.chat_completion(
            ChatRequest(model=None, messages=[ChatMessage(role="user", content="hello")]),
            decision,
        )

    assert seen == [("openai/test", 500)]


@pytest.mark.asyncio
async def test_proxy_skips_disabled_provider_for_remaining_attempts() -> None:
    zai_provider = FailingProvider()
    ollama_provider = SuccessfulProvider()
    zhipu_primary = ModelInfo(name="zhipu/glm-5.2", provider=Provider.ZAI, tier=Tier.T3)
    zhipu_fallback = ModelInfo(name="zhipu/glm-5.1", provider=Provider.ZAI, tier=Tier.T3)
    ollama_fallback = ModelInfo(name="ollama/glm", provider=Provider.OLLAMA, tier=Tier.T3)
    proxy = ProviderProxy(
        {
            Provider.ZAI: zai_provider,
            Provider.OLLAMA: ollama_provider,
        },
    )
    proxy._on_provider_error = lambda failed_model, exc: proxy.disable_provider(
        failed_model.provider
    )
    decision = RoutingDecision(
        primary=zhipu_primary,
        fallbacks=[zhipu_fallback, ollama_fallback],
        score=0.0,
        tier=Tier.T3,
        reason="test",
    )

    response = await proxy.chat_completion(
        ChatRequest(model=None, messages=[ChatMessage(role="user", content="hello")]),
        decision,
    )

    assert response is not None
    assert zai_provider.calls == 1
    assert ollama_provider.calls == 1
