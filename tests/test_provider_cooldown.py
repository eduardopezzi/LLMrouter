"""Tests for provider quota cooldowns and client provider affinity."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmrouter.core.cooldown import (
    CooldownScope,
    ProviderCooldownStore,
    is_model_unavailable_error,
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
        self.calls: list[str] = []
        self.requests: list[ChatRequest] = []

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        self.calls.append(model)
        self.requests.append(request)
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
        self.calls.append(model)
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


def _decision(
    primary: ModelInfo,
    fallbacks: list[ModelInfo],
    *,
    probe_models: tuple[ModelInfo, ...] = (),
) -> RoutingDecision:
    return RoutingDecision(
        primary=primary,
        fallbacks=fallbacks,
        score=0.5,
        tier=primary.tier,
        reason="test",
        probe_models=probe_models,
    )


def test_detects_zai_usage_limit_error() -> None:
    exc = ProviderError(
        'zai returned HTTP 429: {"error":{"message":"Usage limit reached for 5 hour. '
        'Your limit will reset at 2026-07-08 07:41:15"}}',
        status_code=429,
        provider="zai",
    )

    assert is_quota_exhaustion_error(exc) is True


def test_detects_ollama_session_usage_limit_error() -> None:
    exc = ProviderError(
        "ollama returned HTTP 429: you have reached your session usage limit",
        status_code=429,
        provider="ollama",
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
async def test_stream_quota_cooldown_skips_same_provider_fallback() -> None:
    primary = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    same_provider = _model("ollama/kimi-k3:cloud", Provider.OLLAMA)
    cross_provider = _model("zhipu/glm-5.2", Provider.ZAI)
    cooldowns = ProviderCooldownStore(default_seconds=3600)
    ollama = StubProvider(
        "ollama",
        error=ProviderError(
            "ollama returned HTTP 429: you have reached your session usage limit",
            status_code=429,
            provider="ollama",
        ),
    )
    zai = StubProvider("zai")
    proxy = ProviderProxy(
        {Provider.OLLAMA: ollama, Provider.ZAI: zai},
        provider_cooldowns=cooldowns,
    )

    chunks = [
        chunk
        async for chunk in proxy.stream_chat_completion(
            _request(),
            _decision(primary, [same_provider, cross_provider]),
        )
    ]

    assert chunks == [{"choices": [{"delta": {"content": "ok"}}]}]
    assert ollama.calls == [primary.provider_model_name]
    assert zai.calls == [cross_provider.provider_model_name]
    assert cooldowns.cloud_cooldown(Provider.OLLAMA) is not None
    assert cooldowns.provider_cooldown(Provider.OLLAMA) is None


def test_quota_scope_is_provider_for_direct_api() -> None:
    deepseek = _model("deepseek/deepseek-chat", Provider.DEEPSEEK)
    other = _model("deepseek/deepseek-reasoner", Provider.DEEPSEEK)
    cooldowns = ProviderCooldownStore(default_seconds=600)

    entry = cooldowns.record_error(
        deepseek,
        ProviderError("Billing or credits exhausted", status_code=402, provider="deepseek"),
        now=100,
    )

    assert entry is not None
    assert entry.scope == CooldownScope.PROVIDER
    assert cooldowns.cooldown_for_model(other) == entry


def test_ollama_cloud_quota_blocks_cloud_but_not_local_models() -> None:
    failed = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    other_cloud = _model("ollama/kimi-k2.7-code:cloud", Provider.OLLAMA)
    local = _model("ollama/qwen3:8b", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=600)

    entry = cooldowns.record_error(
        failed,
        ProviderError("session usage limit", status_code=429, provider="ollama"),
        now=100,
    )

    assert entry is not None
    assert entry.scope == CooldownScope.CLOUD
    assert cooldowns.cooldown_for_model(failed) == entry
    assert cooldowns.cooldown_for_model(other_cloud) == entry
    assert cooldowns.cooldown_for_model(local) is None


def test_ollama_local_quota_blocks_only_failed_local_model() -> None:
    failed = _model("ollama/qwen3:8b", Provider.OLLAMA)
    other_local = _model("ollama/gemma3:4b", Provider.OLLAMA)
    cloud = _model("ollama/kimi-k2.7-code:cloud", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=600)

    entry = cooldowns.record_error(
        failed,
        ProviderError("token quota exceeded", status_code=429, provider="ollama"),
        now=100,
    )

    assert entry is not None
    assert entry.scope == CooldownScope.MODEL
    assert cooldowns.cooldown_for_model(failed) == entry
    assert cooldowns.cooldown_for_model(other_local) is None
    assert cooldowns.cooldown_for_model(cloud) is None


def test_retired_model_is_removed_from_runtime_routing_without_blocking_provider() -> None:
    retired = _model("ollama/old-model:cloud", Provider.OLLAMA)
    available = _model("ollama/kimi-k2.7-code:cloud", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=600)
    exc = ProviderError(
        "model has been retired and is no longer available",
        status_code=410,
        provider="ollama",
    )

    entry = cooldowns.record_error(retired, exc, now=100)

    assert entry is not None
    assert entry.permanent is True
    assert entry.scope == CooldownScope.MODEL
    assert cooldowns.is_model_retired(retired.name) is True
    assert cooldowns.cooldown_for_model(available) is None
    assert cooldowns.claim_due_probes([retired, available], now=999999) == ()


def test_model_not_found_uses_temporary_model_cooldown() -> None:
    missing = _model("ollama/missing:cloud", Provider.OLLAMA)
    other = _model("ollama/kimi-k2.7-code:cloud", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=600)
    exc = ProviderError("model not found", status_code=404, provider="ollama")

    entry = cooldowns.record_error(missing, exc, now=100)

    assert is_model_unavailable_error(exc) is True
    assert entry is not None
    assert entry.scope == CooldownScope.MODEL
    assert entry.permanent is False
    assert cooldowns.cooldown_for_model(other) is None


def test_half_open_probe_is_single_flight_and_escalates_to_sixty_minutes() -> None:
    failed = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    other_cloud = _model("ollama/kimi-k2.7-code:cloud", Provider.OLLAMA)
    cooldowns = ProviderCooldownStore(default_seconds=600, probe_retry_seconds=3600)
    cooldowns.record_error(
        failed,
        ProviderError("session usage limit", status_code=429, provider="ollama"),
        now=100,
    )

    assert cooldowns.claim_due_probes([failed, other_cloud], now=699) == ()
    assert cooldowns.claim_due_probes([failed, other_cloud], now=700) == (failed,)
    assert cooldowns.claim_due_probes([failed, other_cloud], now=700) == ()

    failed_entry = cooldowns.probe_failed(
        failed,
        ProviderError("still limited", status_code=429, provider="ollama"),
        now=700,
    )

    assert failed_entry is not None
    assert failed_entry.until == 4300
    assert failed_entry.failures == 2
    assert cooldowns.claim_due_probes([failed, other_cloud], now=4299) == ()
    assert cooldowns.claim_due_probes([failed, other_cloud], now=4300) == (failed,)


def test_probe_that_discovers_retired_model_restores_provider_and_removes_model() -> None:
    failed = _model("deepseek/retired", Provider.DEEPSEEK)
    available = _model("deepseek/deepseek-chat", Provider.DEEPSEEK)
    cooldowns = ProviderCooldownStore(default_seconds=600, probe_retry_seconds=3600)
    cooldowns.record_error(
        failed,
        ProviderError("credits exhausted", status_code=402, provider="deepseek"),
        now=100,
    )
    assert cooldowns.claim_due_probes([failed, available], now=700) == (failed,)

    entry = cooldowns.probe_failed(
        failed,
        ProviderError("model removed from catalog", status_code=410, provider="deepseek"),
        now=700,
    )

    assert entry is not None
    assert entry.scope == CooldownScope.MODEL
    assert entry.permanent is True
    assert cooldowns.provider_cooldown(Provider.DEEPSEEK) is None
    assert cooldowns.cooldown_for_model(available) is None


@pytest.mark.asyncio
async def test_due_cooldown_serves_alternative_and_probes_in_parallel() -> None:
    preferred = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    alternative = _model("zhipu/glm-5.2", Provider.ZAI)
    cooldowns = ProviderCooldownStore(default_seconds=600, probe_retry_seconds=3600)
    cooldowns.put_cloud(
        Provider.OLLAMA,
        model_name=preferred.name,
        until=0,
        reason="quota",
    )
    ollama = StubProvider("ollama")
    zai = StubProvider("zai")
    proxy = ProviderProxy(
        {Provider.OLLAMA: ollama, Provider.ZAI: zai},
        provider_cooldowns=cooldowns,
    )
    router = MultiModelRouter(
        ModelRegistry(models=(preferred, alternative)),
        PromptScorer(),
        RoutingStrategy.QUALITY,
        provider_cooldowns=cooldowns,
        client_provider_affinity=False,
    )
    decision = await router.route(_request())

    response = await proxy.chat_completion(_request(), decision)
    await proxy.wait_for_probes()
    next_decision = await router.route(_request())

    assert decision.primary == alternative
    assert decision.probe_models == (preferred,)
    assert response.model == alternative.provider_model_name
    assert zai.calls == [alternative.provider_model_name]
    assert ollama.calls == [preferred.provider_model_name]
    assert ollama.requests[0].prompt_text == "Reply only: OK"
    assert ollama.requests[0].max_tokens == 32
    assert cooldowns.cloud_cooldown(Provider.OLLAMA) is None
    assert next_decision.primary == preferred


@pytest.mark.asyncio
async def test_failed_parallel_probe_reopens_cooldown_for_sixty_minutes() -> None:
    preferred = _model("ollama/deepseek-v4-pro:cloud", Provider.OLLAMA)
    alternative = _model("zhipu/glm-5.2", Provider.ZAI)
    cooldowns = ProviderCooldownStore(default_seconds=600, probe_retry_seconds=3600)
    cooldowns.put_cloud(
        Provider.OLLAMA,
        model_name=preferred.name,
        until=0,
        reason="quota",
    )
    ollama = StubProvider(
        "ollama",
        error=ProviderError("still rate limited", status_code=429, provider="ollama"),
    )
    proxy = ProviderProxy(
        {Provider.OLLAMA: ollama, Provider.ZAI: StubProvider("zai")},
        provider_cooldowns=cooldowns,
    )
    probe_models = cooldowns.claim_due_probes([preferred, alternative])

    response = await proxy.chat_completion(
        _request(),
        _decision(alternative, [], probe_models=probe_models),
    )
    await proxy.wait_for_probes()

    entry = cooldowns.cloud_cooldown(Provider.OLLAMA)
    assert response.model == alternative.provider_model_name
    assert entry is not None
    assert entry.failures == 2
    assert entry.seconds_remaining == pytest.approx(3600, abs=2)


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
