"""Routing policy for expensive providers used only after alternatives."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from llmrouter.config import RoutingConfig
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import ScoringResult
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ModelInfo,
    Provider,
    RoutingConstraints,
    RoutingStrategy,
    Tier,
    Usage,
)
from llmrouter.providers.base import BaseProvider, ProviderError


class FixedScorer:
    def __init__(self, tier: Tier) -> None:
        self.tier = tier

    def score(self, prompt: str) -> ScoringResult:
        return ScoringResult(score=0.7, tier=self.tier, signals={})


def model(name: str, provider: Provider, *, tier: Tier = Tier.T3,
          cost: float = 0.0, rollout: float = 100.0,
          capabilities: frozenset[str] = frozenset({"review"})) -> ModelInfo:
    return ModelInfo(
        name=name, provider=provider, tier=tier, cost_per_1k_input=cost,
        capabilities=capabilities, rollout_percentage=rollout,
    )


def router(
    models: list[ModelInfo], *, tier: Tier = Tier.T3, fallback_count: int = 3
) -> MultiModelRouter:
    return MultiModelRouter(
        ModelRegistry(models=tuple(models)), FixedScorer(tier), RoutingStrategy.COST,
        fallback_count=fallback_count, last_resort_providers=[Provider.DEEPSEEK],
        client_provider_affinity=False, dynamic_benchmark_routing=False,
        intent_routing=False,
    )


def request(name: str | None = None) -> ChatRequest:
    return ChatRequest(
        model=name, messages=[ChatMessage(role="user", content="Review this change")]
    )


@pytest.mark.asyncio
async def test_auto_tries_available_ollama_and_other_providers_before_deepseek() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK, cost=0.01)
    ollama = model("ollama/model", Provider.OLLAMA, cost=0.5)
    zai = model("zhipu/model", Provider.ZAI, cost=0.2)

    decision = await router([deepseek, ollama, zai]).route(request())
    attempts = [decision.primary, *decision.fallbacks]

    assert attempts[0].provider != Provider.DEEPSEEK
    assert ollama in attempts
    assert deepseek in attempts
    assert attempts.index(ollama) < attempts.index(deepseek)
    assert all(m.provider == Provider.DEEPSEEK for m in attempts[attempts.index(deepseek):])


@pytest.mark.asyncio
async def test_auto_uses_eligible_higher_tier_ollama_before_deepseek() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK, tier=Tier.T2)
    ollama = model("ollama/model", Provider.OLLAMA, tier=Tier.T3, cost=0.5)

    decision = await router([deepseek, ollama], tier=Tier.T2).route(request())

    assert decision.primary == ollama
    assert decision.fallbacks[-1] == deepseek


@pytest.mark.asyncio
async def test_deepseek_is_used_when_other_provider_is_unavailable() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK)
    ollama = model("ollama/model", Provider.OLLAMA)
    route = router([deepseek, ollama])
    route.mark_provider_unavailable(Provider.OLLAMA)

    decision = await route.route(request())

    assert decision.primary == deepseek


@pytest.mark.asyncio
async def test_deepseek_is_used_when_others_lack_required_capability() -> None:
    deepseek = model("deepseek/review", Provider.DEEPSEEK)
    ollama = model("ollama/summary", Provider.OLLAMA, capabilities=frozenset({"summary"}))

    decision = await router([deepseek, ollama]).route(
        request(), RoutingConstraints(required_capabilities=frozenset({"review"}))
    )

    assert decision.primary == deepseek
    assert ollama not in decision.fallbacks


@pytest.mark.asyncio
async def test_explicit_deepseek_request_is_respected() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK)
    ollama = model("ollama/model", Provider.OLLAMA)

    decision = await router([deepseek, ollama]).route(request(deepseek.name))

    assert decision.primary == deepseek


@pytest.mark.asyncio
async def test_explicit_other_model_falls_back_to_deepseek_last() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK, cost=0.01)
    ollama = model("ollama/model", Provider.OLLAMA, cost=0.5)
    zai = model("zhipu/model", Provider.ZAI, cost=0.2)

    decision = await router([deepseek, ollama, zai]).route(request(ollama.name))

    assert decision.primary == ollama
    assert decision.fallbacks == [zai, deepseek]


@pytest.mark.asyncio
async def test_proxy_uses_deepseek_only_after_alternatives_fail() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK, cost=0.01)
    ollama = model("ollama/model", Provider.OLLAMA, cost=0.5)
    zai = model("zhipu/model", Provider.ZAI, cost=0.2)
    attempts: list[str] = []

    class StubProvider(BaseProvider):
        def __init__(self, provider: Provider) -> None:
            super().__init__(provider.value)
            self.provider = provider

        async def chat_completion(self, req: ChatRequest, name: str) -> ChatResponse:
            attempts.append(self.provider.value)
            if self.provider != Provider.DEEPSEEK:
                raise ProviderError("unavailable", status_code=503, provider=self.provider.value)
            return ChatResponse(
                id="test", model=name, choices=[], usage=Usage(),
                finish_reason=FinishReason.STOP,
            )

        async def stream_completion(
            self, req: ChatRequest, name: str
        ) -> AsyncIterator[dict[str, object]]:
            yield {"model": name}

    decision = await router([deepseek, ollama, zai]).route(request())
    proxy = ProviderProxy({
        p: StubProvider(p) for p in [Provider.DEEPSEEK, Provider.OLLAMA, Provider.ZAI]
    })

    response = await proxy.chat_completion(request(), decision)

    assert response.model == "flash"
    assert attempts == ["zai", "ollama", "deepseek"]


@pytest.mark.asyncio
async def test_rollout_zero_alternative_precedes_last_resort() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK)
    ollama = model("ollama/emergency", Provider.OLLAMA, rollout=0)

    decision = await router([deepseek, ollama]).route(request())

    assert decision.primary == deepseek
    assert ollama in decision.fallbacks


@pytest.mark.asyncio
async def test_last_resort_respects_total_fallback_limit() -> None:
    deepseek = model("deepseek/flash", Provider.DEEPSEEK, cost=0.01)
    ollama = model("ollama/model", Provider.OLLAMA, cost=0.5)
    zai = model("zhipu/model", Provider.ZAI, cost=0.2)

    decision = await router([deepseek, ollama, zai], fallback_count=1).route(request())

    assert decision.primary == zai
    assert decision.fallbacks == [deepseek]


def test_last_resort_configuration_is_optional_and_validates_provider() -> None:
    assert RoutingConfig().last_resort_providers == []
    config = RoutingConfig(last_resort_providers=["deepseek"])
    assert config.last_resort_providers == [Provider.DEEPSEEK]
    with pytest.raises(ValueError):
        RoutingConfig(last_resort_providers=["unknown-provider"])
