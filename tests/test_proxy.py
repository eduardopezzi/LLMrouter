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
