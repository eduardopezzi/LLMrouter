"""Tests for provider quota cooldowns and client provider affinity."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmrouter.core.cooldown import (
    ProviderCooldownStore,
    is_quota_exhaustion_error,
    quota_reset_timestamp,
)
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ModelInfo,
    Provider,
    RoutingDecision,
    RoutingStrategy,
    Tier,
    Usage,
)
from llmrouter.providers.base import BaseProvider, ProviderError

UTC = timezone.utc  # noqa: UP017 - keep Python 3.10 compatibility.


class StubProvider(BaseProvider):
    def __init__(self, name: str, *, error: ProviderError | None = None) -> None:
        super().__init__(name)
        self.error = error

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        if self.error is not None:
            raise self.error
        return ChatResponse(
            id="id",
            model=model,
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason=FinishReason.STOP,
        )

    async def stream_completion(self, request: ChatRequest, model: str):
        if self.error is not None:
            raise self.error
        yield {"choices": [{"delta": {"content": "ok"}}]}


def _model(name: str, provider: Provider, tier: Tier = Tier.T3) -> ModelInfo:
    return ModelInfo(name=name, provider=provider, tier=tier)


def _request(*, client_ip: str = "10.0.0.1") -> ChatRequest:
    return ChatRequest(
        model=None,
        messages=[ChatMessage(role="user", content="review this code architecture")],
        extra={"_llmrouter_client_ip": client_ip, "_llmrouter_client_id": client_ip},
    )


def _decision(primary: ModelInfo, fallbacks: list[ModelInfo]) -> RoutingDecision:
    return RoutingDecision(
        primary=primary,
        fallbacks=fallbacks,
        score=0.5,
        tier=primary.tier,
        reason="test",
    )


def test_detects_zai_usage_limit_error() -> None:
    exc = ProviderError(
        'zai returned HTTP 429: {"error":{"message":"Usage limit reached for 5 hour. '
        'Your limit will reset at 2026-07-08 07:41:15"}}',
        status_code=429,
        provider="zai",
    )

    assert is_quota_exhaustion_error(exc) is True


def test_parses_reset_timestamp_as_utc() -> None:
    reset = quota_reset_timestamp(
        "Usage limit reached. Your limit will reset at 2026-07-08 07:41:15",
        default_seconds=300,
    )

    expected = datetime(2026, 7, 8, 7, 41, 15, tzinfo=UTC).timestamp()
    assert reset == pytest.approx(expected, abs=1)


@pytest.mark.asyncio
async def test_proxy_records_quota_cooldown_and_uses_fallback() -> None:
    zai = _model("zhipu/glm-5.2", Provider.ZAI)
    ollama = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=3600)
    proxy = ProviderProxy(
        {
            Provider.ZAI: StubProvider(
                "zai",
                error=ProviderError(
                    "Usage limit reached for 5 hour",
                    status_code=429,
                    provider="zai",
                ),
            ),
            Provider.OLLAMA: StubProvider("ollama"),
        },
        provider_cooldowns=cooldowns,
    )

    response = await proxy.chat_completion(_request(), _decision(zai, [ollama]))

    assert response.model == ollama.provider_model_name
    assert cooldowns.provider_cooldown(Provider.ZAI) is not None
    assert Provider.ZAI not in proxy.providers


@pytest.mark.asyncio
async def test_router_skips_provider_in_cooldown() -> None:
    zai = _model("zhipu/glm-5.2", Provider.ZAI)
    ollama = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=3600)
    cooldowns.put_provider(Provider.ZAI, until=9999999999, reason="quota")
    router = MultiModelRouter(
        ModelRegistry(models=(zai, ollama)),
        PromptScorer(),
        RoutingStrategy.QUALITY,
        provider_cooldowns=cooldowns,
    )

    decision = await router.route(_request())

    assert decision.primary.provider == Provider.OLLAMA


def test_client_provider_affinity_can_choose_different_providers() -> None:
    providers_seen: set[Provider] = set()
    models = [
        _model("zhipu/glm-5.2", Provider.ZAI),
        _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA),
        _model("deepseek/deepseek-chat", Provider.DEEPSEEK),
    ]
    router = MultiModelRouter(
        ModelRegistry(models=tuple(models)),
        PromptScorer(),
        RoutingStrategy.QUALITY,
        client_provider_affinity=True,
    )

    for index in range(30):
        ordered = router._apply_client_provider_affinity(
            models,
            _request(client_ip=f"10.0.0.{index}"),
            constraints=type("Constraints", (), {"preferred_provider": None})(),
        )
        providers_seen.add(ordered[0].provider)

    assert len(providers_seen) >= 2
