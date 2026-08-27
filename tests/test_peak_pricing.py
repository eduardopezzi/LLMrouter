from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from llmrouter.config import Settings
from llmrouter.core.peak_pricing import PeakPricingPriorityPolicy, ProviderPricingRule
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import ChatMessage, ChatRequest, ModelInfo, Provider, Tier
from llmrouter.runtime import _build_peak_pricing_policy

UTC = timezone.utc  # noqa: UP017 - test suite supports Python 3.10.


def _deepseek_rule() -> ProviderPricingRule:
    return ProviderPricingRule(
        provider=Provider.DEEPSEEK,
        timezone_name="Asia/Shanghai",
        off_peak_start=time(0, 30),
        off_peak_end=time(8, 30),
        weekend_off_peak_from=date(2026, 8, 23),
    )


@pytest.mark.parametrize(
    ("instant", "is_peak"),
    [
        (datetime(2026, 8, 24, 0, 29, tzinfo=UTC), False),  # 08:29 Monday
        (datetime(2026, 8, 24, 0, 30, tzinfo=UTC), True),  # 08:30 Monday
        (datetime(2026, 8, 24, 16, 29, tzinfo=UTC), True),  # 00:29 Tuesday
        (datetime(2026, 8, 24, 16, 30, tzinfo=UTC), False),  # 00:30 Tuesday
    ],
)
def test_deepseek_weekday_peak_window_uses_beijing_time(
    instant: datetime,
    is_peak: bool,
) -> None:
    assert _deepseek_rule().is_peak(instant) is is_peak


def test_deepseek_weekend_is_off_peak_all_day_after_new_rule() -> None:
    sunday_noon_beijing = datetime(2026, 8, 23, 4, 0, tzinfo=UTC)

    assert _deepseek_rule().is_peak(sunday_noon_beijing) is False


def test_deepseek_weekend_used_daily_window_before_new_rule() -> None:
    saturday_noon_beijing = datetime(2026, 8, 22, 4, 0, tzinfo=UTC)

    assert _deepseek_rule().is_peak(saturday_noon_beijing) is True


def test_peak_policy_demotes_deepseek_but_keeps_it_available() -> None:
    deepseek = ModelInfo("deepseek/chat", Provider.DEEPSEEK, Tier.T3)
    ollama = ModelInfo("ollama/kimi", Provider.OLLAMA, Tier.T3)
    policy = PeakPricingPriorityPolicy(
        [_deepseek_rule()],
        clock=lambda: datetime(2026, 8, 24, 4, 0, tzinfo=UTC),
    )

    assert policy.prioritize([deepseek, ollama]) == [ollama, deepseek]


def test_off_peak_policy_preserves_existing_order() -> None:
    deepseek = ModelInfo("deepseek/chat", Provider.DEEPSEEK, Tier.T3)
    ollama = ModelInfo("ollama/kimi", Provider.OLLAMA, Tier.T3)
    policy = PeakPricingPriorityPolicy(
        [_deepseek_rule()],
        clock=lambda: datetime(2026, 8, 24, 17, 0, tzinfo=UTC),
    )

    assert policy.prioritize([deepseek, ollama]) == [deepseek, ollama]


@pytest.mark.asyncio
async def test_auto_route_demotes_deepseek_during_peak() -> None:
    deepseek = ModelInfo(
        "deepseek/reviewer",
        Provider.DEEPSEEK,
        Tier.T3,
        priority=1,
        capabilities=frozenset({"review"}),
    )
    ollama = ModelInfo(
        "ollama/reviewer",
        Provider.OLLAMA,
        Tier.T3,
        priority=2,
        capabilities=frozenset({"review"}),
    )
    policy = PeakPricingPriorityPolicy(
        [_deepseek_rule()],
        clock=lambda: datetime(2026, 8, 24, 4, 0, tzinfo=UTC),
    )
    router = MultiModelRouter(
        ModelRegistry(models=(deepseek, ollama)),
        PromptScorer(),
        fallback_count=1,
        client_provider_affinity=False,
        dynamic_benchmark_routing=False,
        peak_pricing_policy=policy,
    )
    request = ChatRequest(
        model=None,
        messages=[ChatMessage(role="user", content="Review this migration architecture")],
    )

    decision = await router.route(request)

    assert decision.primary == ollama
    assert decision.fallbacks == [deepseek]


@pytest.mark.asyncio
async def test_explicit_deepseek_selection_is_not_demoted_during_peak() -> None:
    deepseek = ModelInfo("deepseek/reviewer", Provider.DEEPSEEK, Tier.T3, priority=1)
    ollama = ModelInfo("ollama/reviewer", Provider.OLLAMA, Tier.T3, priority=2)
    policy = PeakPricingPriorityPolicy(
        [_deepseek_rule()],
        clock=lambda: datetime(2026, 8, 24, 4, 0, tzinfo=UTC),
    )
    router = MultiModelRouter(
        ModelRegistry(models=(deepseek, ollama)),
        PromptScorer(),
        peak_pricing_policy=policy,
    )
    request = ChatRequest(
        model=deepseek.name,
        messages=[ChatMessage(role="user", content="Review this")],
    )

    decision = await router.route(request)

    assert decision.primary == deepseek


def test_runtime_builds_default_deepseek_pricing_policy() -> None:
    policy = _build_peak_pricing_policy(Settings())

    assert policy is not None


def test_runtime_can_disable_peak_pricing_policy() -> None:
    settings = Settings()
    settings.routing.deepseek_pricing.enabled = False

    assert _build_peak_pricing_policy(settings) is None
