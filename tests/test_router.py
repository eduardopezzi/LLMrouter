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
    RoutingConstraints,
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


@pytest.mark.asyncio
async def test_cost_strategy_prefers_nvidia_then_zai_then_ollama_on_equal_cost() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="ollama/reviewer",
                provider=Provider.OLLAMA,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
                priority=1,
            ),
            ModelInfo(
                name="zhipu/reviewer",
                provider=Provider.ZAI,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
                priority=30,
            ),
            ModelInfo(
                name="nvidia/reviewer",
                provider=Provider.NVIDIA,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
                priority=20,
            ),
        )
    )
    request = ChatRequest(
        model=None,
        messages=[ChatMessage(role="user", content="Review this migration architecture.")],
    )
    constraints = RoutingConstraints(required_capabilities=frozenset({"review"}))
    router = MultiModelRouter(registry, PromptScorer(), RoutingStrategy.COST)

    decision = await router.route(request, constraints)

    assert decision.primary.name == "nvidia/reviewer"
    assert [model.name for model in decision.fallbacks] == [
        "zhipu/reviewer",
        "ollama/reviewer",
    ]


@pytest.mark.asyncio
async def test_cost_strategy_uses_configured_provider_order_on_equal_cost() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="ollama/reviewer",
                provider=Provider.OLLAMA,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
                priority=1,
            ),
            ModelInfo(
                name="nvidia/reviewer",
                provider=Provider.NVIDIA,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
                priority=20,
            ),
        )
    )
    request = ChatRequest(
        model=None,
        messages=[ChatMessage(role="user", content="Review this migration architecture.")],
    )
    constraints = RoutingConstraints(required_capabilities=frozenset({"review"}))
    router = MultiModelRouter(
        registry,
        PromptScorer(),
        RoutingStrategy.COST,
        provider_cost_order=["ollama", "nvidia"],
    )

    decision = await router.route(request, constraints)

    assert decision.primary.name == "ollama/reviewer"
    assert [model.name for model in decision.fallbacks] == ["nvidia/reviewer"]
