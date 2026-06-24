from __future__ import annotations

import pytest

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingStrategy,
    Tier,
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(
                name="cheap",
                provider=Provider.OPENAI,
                tier=Tier.T1,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.01,
            ),
            ModelInfo(
                name="code",
                provider=Provider.OPENAI,
                tier=Tier.T3,
                cost_per_1k_input=1.0,
                cost_per_1k_output=1.0,
                capabilities=frozenset({"code"}),
                priority=1,
            ),
        )
    )


@pytest.mark.asyncio
async def test_router_uses_explicit_registered_model() -> None:
    request = ChatRequest(
        model="cheap",
        messages=[ChatMessage(role="user", content="hello")],
    )
    router = MultiModelRouter(_registry(), PromptScorer(), RoutingStrategy.BALANCED)

    decision = await router.route(request)

    assert decision.primary.name == "cheap"
    assert decision.tier == Tier.T1
    assert "Explicit model selection" in decision.reason


@pytest.mark.asyncio
async def test_router_scores_code_prompt_into_higher_tier() -> None:
    request = ChatRequest(
        model=None,
        messages=[
            ChatMessage(
                role="user",
                content="Debug and refactor this implementation:\n```python\ndef x(): pass\n```",
            )
        ],
    )
    router = MultiModelRouter(_registry(), PromptScorer(), RoutingStrategy.QUALITY)

    decision = await router.route(request)

    assert decision.primary.name == "code"
    assert decision.primary.tier == Tier.T3
    assert decision.score > 0
    assert decision.tier == Tier.T2
