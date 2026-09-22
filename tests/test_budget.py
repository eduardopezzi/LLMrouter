"""Tests for the budget manager (Fase 5 / B1 - DEVELOPMENT_PLAN_TDD.md:209-217).

Covers: allow within limit, hard block (daily and monthly), soft warning,
tenant isolation, daily/monthly period resets via injectable clock,
persistence across manager instances, zero-cost structured warning,
check() read-only idempotency, no-limits enforcement bypass, and
concurrent record_usage without lost increments.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llmrouter.core.budget import (
    DEFAULT_PROJECT_ID,
    DEFAULT_USER_ID,
    BudgetLimits,
    BudgetManager,
    estimate_cost,
)


class FakeClock:
    """Injectable clock returning a mutable ``now`` (UTC aware)."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _clock_at(year: int, month: int, day: int, hour: int = 12) -> FakeClock:
    return FakeClock(datetime(year, month, day, hour, 0, 0, tzinfo=UTC))


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "budget_test.db")


def _manager(db_path: str, clock: FakeClock | None = None) -> BudgetManager:
    return BudgetManager(db_path, clock=clock)


# ---------------------------------------------------------------------------
# estimate_cost helper (mirrors proxy._estimate_cost formula)
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_matches_proxy_formula(self) -> None:
        # (input/1000)*in_cost + (output/1000)*out_cost
        assert estimate_cost(0.5, 1.5, 1000, 2000) == pytest.approx(0.5 + 3.0)

    def test_zero_tokens_zero_cost(self) -> None:
        assert estimate_cost(10.0, 20.0, 0, 0) == 0.0

    def test_zero_price_zero_cost(self) -> None:
        assert estimate_cost(0.0, 0.0, 5000, 5000) == 0.0


# ---------------------------------------------------------------------------
# 1. Within limit -> allowed, no warning
# ---------------------------------------------------------------------------


async def test_within_limit_allows_without_warning(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="hard"))
    await mgr.record_usage("p1", "u1", 5.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=1.0)

    assert decision.allowed is True
    assert decision.reason is None
    assert decision.warning is None


# ---------------------------------------------------------------------------
# 2. Hard mode exceeded (daily OR monthly) -> denied with reason
# ---------------------------------------------------------------------------


async def test_hard_daily_exceeded_denies(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="hard"))
    await mgr.record_usage("p1", "u1", 9.5)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=1.0)

    assert decision.allowed is False
    assert decision.reason is not None
    assert "daily budget exceeded" in decision.reason


async def test_hard_monthly_exceeded_denies(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits(
        "p1", "u1", BudgetLimits(monthly_limit_usd=20.0, mode="hard")
    )
    await mgr.record_usage("p1", "u1", 15.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=10.0)

    assert decision.allowed is False
    assert decision.reason is not None
    assert "monthly budget exceeded" in decision.reason


async def test_hard_at_exact_limit_still_allowed(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="hard"))
    await mgr.record_usage("p1", "u1", 9.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=1.0)

    assert decision.allowed is True  # spend + estimate == limit: not exceeded


# ---------------------------------------------------------------------------
# 3. Soft mode exceeded -> allowed with warning, never blocks
# ---------------------------------------------------------------------------


async def test_soft_exceeded_warns_but_allows(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="soft"))
    await mgr.record_usage("p1", "u1", 9.5)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=5.0)

    assert decision.allowed is True
    assert decision.reason is None
    assert decision.warning is not None
    assert "daily budget exceeded" in decision.warning


async def test_soft_monthly_exceeded_warns_but_allows(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(monthly_limit_usd=1.0, mode="soft"))
    await mgr.record_usage("p1", "u1", 0.5)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=2.0)

    assert decision.allowed is True
    assert decision.warning is not None
    assert "monthly budget exceeded" in decision.warning


# ---------------------------------------------------------------------------
# 4. Tenants are independent
# ---------------------------------------------------------------------------


async def test_tenants_independent(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    limits = BudgetLimits(daily_limit_usd=10.0, mode="hard")
    await mgr.set_limits("projA", "userA", limits)
    await mgr.set_limits("projB", "userB", limits)
    await mgr.record_usage("projA", "userA", 9.5)

    decision_a = await mgr.check("projA", "userA", estimated_cost_usd=1.0)
    decision_b = await mgr.check("projB", "userB", estimated_cost_usd=1.0)

    assert decision_a.allowed is False
    assert decision_b.allowed is True
    assert decision_b.warning is None

    usage_b = await mgr.get_usage("projB", "userB")
    assert usage_b.daily_spent_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Period resets via injectable clock
# ---------------------------------------------------------------------------


async def test_daily_reset_on_day_rollover(db_path: str) -> None:
    clock = _clock_at(2026, 9, 20)
    mgr = _manager(db_path, clock)
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="hard"))
    await mgr.record_usage("p1", "u1", 9.0)

    clock.now = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    usage = await mgr.get_usage("p1", "u1")
    assert usage.period_day == "2026-09-21"
    assert usage.daily_spent_usd == pytest.approx(0.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=5.0)
    assert decision.allowed is True


async def test_monthly_reset_on_month_rollover(db_path: str) -> None:
    clock = _clock_at(2026, 8, 31)
    mgr = _manager(db_path, clock)
    await mgr.set_limits("p1", "u1", BudgetLimits(monthly_limit_usd=30.0, mode="hard"))
    await mgr.record_usage("p1", "u1", 25.0)

    clock.now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)

    usage = await mgr.get_usage("p1", "u1")
    assert usage.period_month == "2026-09"
    assert usage.monthly_spent_usd == pytest.approx(0.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=20.0)
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 6. Persistence across manager instances on the same db_path
# ---------------------------------------------------------------------------


async def test_persistence_across_managers(db_path: str) -> None:
    mgr1 = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr1.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="hard"))
    await mgr1.record_usage("p1", "u1", 9.5)

    mgr2 = BudgetManager(db_path)  # real clock, same day in UTC
    usage = await mgr2.get_usage("p1", "u1")

    assert usage.project_id == "p1"
    assert usage.user_id == "u1"
    assert usage.daily_limit_usd == pytest.approx(10.0)
    assert usage.mode == "hard"
    assert usage.daily_spent_usd == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# 7. Zero-cost usage warning (once per period per tenant)
# ---------------------------------------------------------------------------


async def test_zero_cost_warning_emitted_once_per_period(
    db_path: str, caplog: Any
) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0))

    with caplog.at_level(logging.WARNING, logger="llmrouter.budget"):
        await mgr.record_usage("p1", "u1", 0.0, cost_known=False)
        await mgr.record_usage("p1", "u1", 0.0, cost_known=False)

    warnings = [r for r in caplog.records if "budget_zero_cost_usage" in r.message]
    assert len(warnings) == 1
    assert "p1" in warnings[0].message
    assert "u1" in warnings[0].message


async def test_zero_cost_known_does_not_warn(db_path: str, caplog: Any) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))

    with caplog.at_level(logging.WARNING, logger="llmrouter.budget"):
        await mgr.record_usage("p1", "u1", 0.0, cost_known=True)

    assert not [r for r in caplog.records if "budget_zero_cost_usage" in r.message]


async def test_zero_cost_warning_resets_next_period(
    db_path: str, caplog: Any
) -> None:
    clock = _clock_at(2026, 9, 20)
    mgr = _manager(db_path, clock)

    with caplog.at_level(logging.WARNING, logger="llmrouter.budget"):
        await mgr.record_usage("p1", "u1", 0.0, cost_known=False)
        clock.now = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
        await mgr.record_usage("p1", "u1", 0.0, cost_known=False)

    warnings = [r for r in caplog.records if "budget_zero_cost_usage" in r.message]
    assert len(warnings) == 2  # one per period


# ---------------------------------------------------------------------------
# 8. check() must not mutate usage (read-only pre-flight)
# ---------------------------------------------------------------------------


async def test_check_does_not_mutate_usage(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=10.0, mode="hard"))
    await mgr.record_usage("p1", "u1", 5.0)

    first = await mgr.check("p1", "u1", estimated_cost_usd=2.0)
    second = await mgr.check("p1", "u1", estimated_cost_usd=2.0)
    usage = await mgr.get_usage("p1", "u1")

    assert first == second
    assert usage.daily_spent_usd == pytest.approx(5.0)
    assert usage.monthly_spent_usd == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 9. No limits configured -> always allowed
# ---------------------------------------------------------------------------


async def test_no_limits_row_always_allowed(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.record_usage("p1", "u1", 1_000_000.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=1_000.0)

    assert decision.allowed is True
    assert decision.reason is None
    assert decision.warning is None


async def test_none_limits_always_allowed(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits())  # both limits None
    await mgr.record_usage("p1", "u1", 500.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=500.0)

    assert decision.allowed is True
    assert decision.warning is None


# ---------------------------------------------------------------------------
# 10. Concurrency: simultaneous record_usage must not lose increments
# ---------------------------------------------------------------------------


async def test_concurrent_record_usage_no_lost_increment(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits("p1", "u1", BudgetLimits(daily_limit_usd=100.0, mode="hard"))

    await asyncio.gather(
        mgr.record_usage("p1", "u1", 0.5),
        mgr.record_usage("p1", "u1", 0.5),
    )

    usage = await mgr.get_usage("p1", "u1")
    assert usage.daily_spent_usd == pytest.approx(1.0)
    assert usage.monthly_spent_usd == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tenant fallback + usage snapshot shape
# ---------------------------------------------------------------------------


async def test_tenant_fallback_to_default_ids(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.record_usage(None, "", 3.0)  # type: ignore[arg-type]

    usage = await mgr.get_usage(DEFAULT_PROJECT_ID, DEFAULT_USER_ID)
    assert usage.project_id == DEFAULT_PROJECT_ID
    assert usage.user_id == DEFAULT_USER_ID
    assert usage.daily_spent_usd == pytest.approx(3.0)


async def test_usage_snapshot_shape(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits(
        "p1", "u1", BudgetLimits(daily_limit_usd=10.0, monthly_limit_usd=100.0)
    )
    usage = await mgr.get_usage("p1", "u1")

    assert usage.period_day == "2026-09-21"
    assert usage.period_month == "2026-09"
    assert usage.daily_limit_usd == pytest.approx(10.0)
    assert usage.monthly_limit_usd == pytest.approx(100.0)
    assert usage.mode == "soft"
    assert usage.daily_spent_usd == pytest.approx(0.0)
    assert usage.monthly_spent_usd == pytest.approx(0.0)


async def test_remaining_usd_reflects_headroom(db_path: str) -> None:
    mgr = _manager(db_path, _clock_at(2026, 9, 21))
    await mgr.set_limits(
        "p1", "u1",
        BudgetLimits(daily_limit_usd=10.0, monthly_limit_usd=100.0, mode="hard"),
    )
    await mgr.record_usage("p1", "u1", 4.0)

    decision = await mgr.check("p1", "u1", estimated_cost_usd=1.0)

    assert decision.allowed is True
    assert decision.remaining_usd == pytest.approx(6.0)  # min(10-4, 100-4)
