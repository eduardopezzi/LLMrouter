"""Tests for model health tracking and adaptive routing integration."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import deque
from pathlib import Path

import pytest

from llmrouter.core.health import (
    HealthWeights,
    InMemoryHealthStore,
    ModelHealthTracker,
    SQLiteHealthStore,
    _aggregate,
    _percentile,
    _score_cost,
    _score_error_rate,
    _score_latency,
    _score_quality,
)
from llmrouter.core.router import (
    BalancedStrategy,
    CostStrategy,
    LatencyStrategy,
    MultiModelRouter,
    QualityStrategy,
)
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import ChatMessage, ChatRequest, ModelInfo, Provider, Tier


@pytest.fixture
def tracker() -> ModelHealthTracker:
    return ModelHealthTracker(window_minutes=15)


@pytest.mark.asyncio
async def test_record_success_creates_health(tracker: ModelHealthTracker) -> None:
    await tracker.record_success("m1", latency_ms=100, cost_usd=0.001, quality=4.5)
    health = await tracker.get_health("m1")
    assert health.model_name == "m1"
    assert health.p50_ms == 100.0
    assert health.p95_ms == 100.0
    assert health.avg_latency_ms == 100.0
    assert health.avg_cost_usd == 0.001
    assert health.avg_quality == 4.5
    assert health.request_count == 1


@pytest.mark.asyncio
async def test_error_rate_computation(tracker: ModelHealthTracker) -> None:
    await tracker.record_success("m2", latency_ms=200, cost_usd=0.001)
    await tracker.record_success("m2", latency_ms=200, cost_usd=0.001)
    await tracker.record_error("m2", error_type="timeout")
    health = await tracker.get_health("m2")
    assert health.error_rate == pytest.approx(1 / 3)
    assert health.request_count == 3


@pytest.mark.asyncio
async def test_health_score_falls_with_errors(tracker: ModelHealthTracker) -> None:
    await tracker.record_success("m3", latency_ms=300, cost_usd=0.001, quality=3.0)
    score1 = await tracker.health_score("m3")
    await tracker.record_error("m3")
    score2 = await tracker.health_score("m3")
    assert score2.score < score1.score


@pytest.mark.asyncio
async def test_sliding_window_expires_old_events() -> None:
    # Use a tracker with a fixed window start so we can seed old events safely.
    from llmrouter.core.health import HealthEvent, InMemoryHealthStore, ModelHealthTracker

    store = InMemoryHealthStore()
    now = 1_000_000.0
    old_event = now - 20 * 60  # 20 minutes old

    async with store._lock:
        store._events["m4"] = deque(
            [
                HealthEvent("m4", old_event, True, 100, 0.0, 0.0),
                HealthEvent("m4", now, True, 200, 0.0, 0.0),
            ]
        )

    tracker = ModelHealthTracker(store=store, window_minutes=15)
    # Patch time used by get_health to match seeded events
    original_time = time.time

    def _fixed_time():
        return now

    time.time = _fixed_time  # type: ignore[assignment]
    try:
        health = await tracker.get_health("m4")
    finally:
        time.time = original_time  # type: ignore[assignment]
    assert health.avg_latency_ms == 200.0
    assert health.request_count == 1


@pytest.mark.asyncio
async def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    store = SQLiteHealthStore(db_path=str(db_path), ttl_minutes=60)
    await store.record_success("s1", latency_ms=123, cost_usd=0.002, quality=4.0)
    await store.record_error("s1", error_type="http_500")
    health = await store.get_health("s1", window_minutes=15, now_ts=time.time())
    assert health.request_count == 2
    assert health.error_rate == 0.5
    assert health.avg_latency_ms == 123.0


def test_percentile_calculation() -> None:
    values = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    assert _percentile(values, 0.50) == 30.0
    assert _percentile(values, 0.95) == 48.0  # linear interpolation: index=3.8
    assert _percentile([], 0.50) == 0.0
    assert _percentile([42.0], 0.99) == 42.0


def test_score_helpers() -> None:
    assert _score_latency(0) == 1.0
    assert _score_latency(15000) == 0.0
    assert _score_error_rate(0.0) == 1.0
    assert _score_error_rate(1.0) == 0.0
    assert _score_quality(0.0) == 0.5
    assert _score_quality(5.0) == 1.0
    assert _score_cost(0.0) == 1.0
    assert _score_cost(0.10) == 0.0


def _model(name: str, tier: Tier, provider: Provider, cost: float = 0.0) -> ModelInfo:
    return ModelInfo(
        name=name,
        provider=provider,
        tier=tier,
        cost_per_1k_input=cost,
        cost_per_1k_output=cost,
        priority=10,
    )


@pytest.mark.asyncio
async def test_cost_strategy_prefers_healthy_model_when_cost_equal() -> None:
    tracker = ModelHealthTracker(window_minutes=15)
    # Equal catalog cost: health becomes the tie-breaker
    await tracker.record_success("m1", latency_ms=12000, cost_usd=0.001)
    await tracker.record_error("m1")
    await tracker.record_success("m2", latency_ms=500, cost_usd=0.001, quality=4.5)

    models = [
        _model("m1", Tier.T1, Provider.OLLAMA, cost=0.001),
        _model("m2", Tier.T1, Provider.OLLAMA, cost=0.001),
    ]
    strategy = CostStrategy()
    strategy.set_health_tracker(tracker)
    ordered = strategy.select(models, None)  # type: ignore[arg-type]
    assert ordered[0].name == "m2"


@pytest.mark.asyncio
async def test_latency_strategy_prefers_lower_p95() -> None:
    tracker = ModelHealthTracker(window_minutes=15)
    await tracker.record_success("slow", latency_ms=8000, cost_usd=0.001)
    await tracker.record_success("slow", latency_ms=9000, cost_usd=0.001)
    await tracker.record_success("fast", latency_ms=500, cost_usd=0.001)
    await tracker.record_success("fast", latency_ms=600, cost_usd=0.001)

    models = [
        _model("slow", Tier.T1, Provider.OLLAMA, cost=0.001),
        _model("fast", Tier.T1, Provider.OLLAMA, cost=0.001),
    ]
    strategy = LatencyStrategy()
    strategy.set_health_tracker(tracker)
    ordered = strategy.select(models, None)  # type: ignore[arg-type]
    assert ordered[0].name == "fast"


@pytest.mark.asyncio
async def test_balanced_strategy_penalizes_sick_model() -> None:
    tracker = ModelHealthTracker(window_minutes=15)
    await tracker.record_success("a", latency_ms=500, cost_usd=0.001, quality=4.5)
    await tracker.record_error("b")
    await tracker.record_error("b")

    models = [
        _model("a", Tier.T1, Provider.OLLAMA, cost=0.001),
        _model("b", Tier.T1, Provider.OLLAMA, cost=0.001),
    ]
    strategy = BalancedStrategy()
    strategy.set_health_tracker(tracker)
    ordered = strategy.select(models, None)  # type: ignore[arg-type]
    assert ordered[0].name == "a"


@pytest.mark.asyncio
async def test_quality_strategy_tie_break_by_health() -> None:
    tracker = ModelHealthTracker(window_minutes=15)
    await tracker.record_success("x", latency_ms=200, cost_usd=0.0, quality=5.0)
    await tracker.record_error("y")

    models = [
        _model("x", Tier.T2, Provider.OLLAMA),
        _model("y", Tier.T2, Provider.OLLAMA),
    ]
    strategy = QualityStrategy()
    strategy.set_health_tracker(tracker)
    models_sorted = strategy.select(models, None)  # type: ignore[arg-type]
    assert models_sorted[0].name == "x"


@pytest.mark.asyncio
async def test_router_without_tracker_still_works() -> None:
    from llmrouter.core.registry import ModelRegistry

    registry = ModelRegistry()
    registry = registry.add(_model("only", Tier.T1, Provider.OLLAMA))
    router = MultiModelRouter(registry, PromptScorer(), fallback_count=0)
    decision = await router.route(
        ChatRequest(model=None, messages=[ChatMessage(role="user", content="hello")])
    )
    assert decision.primary.name == "only"


@pytest.mark.asyncio
async def test_router_with_tracker_uses_health_for_routing() -> None:
    from llmrouter.core.registry import ModelRegistry

    registry = ModelRegistry()
    registry = registry.add(_model("sick", Tier.T1, Provider.OLLAMA, cost=0.001))
    registry = registry.add(_model("well", Tier.T1, Provider.OLLAMA, cost=0.001))

    tracker = ModelHealthTracker(window_minutes=15)
    await tracker.record_success("well", latency_ms=200, cost_usd=0.0001, quality=4.5)
    await tracker.record_error("sick")

    router = MultiModelRouter(
        registry,
        PromptScorer(),
        fallback_count=0,
        health_tracker=tracker,
    )
    decision = await router.route(
        ChatRequest(model=None, messages=[ChatMessage(role="user", content="hello")])
    )
    assert decision.primary.name == "well"
