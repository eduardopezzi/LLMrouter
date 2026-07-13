"""Tests for the operational metrics collector (Fase 3)."""

from __future__ import annotations

import pytest

from llmrouter.core.stats import (
    LatencyStats,
    MetricsCollector,
    ProviderErrorStats,
    StatsSnapshot,
    TierDistribution,
    _percentile,
)


class TestPercentile:
    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 50) == 0.0
        assert _percentile([], 95) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([100.0], 50) == 100.0
        assert _percentile([100.0], 95) == 100.0

    def test_p50_p95(self) -> None:
        samples = [float(i) for i in range(1, 101)]  # 1..100
        assert _percentile(samples, 50) == pytest.approx(50.5)
        assert _percentile(samples, 95) == pytest.approx(95.05)


class TestLatencyStats:
    def test_empty(self) -> None:
        stats = LatencyStats()
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.count == 0

    def test_record_and_percentiles(self) -> None:
        stats = LatencyStats()
        for i in range(1, 11):
            stats.record(float(i * 10))
        assert stats.count == 10
        assert stats.p50 == 55.0
        assert stats.p95 == pytest.approx(95.5)

    def test_to_dict(self) -> None:
        stats = LatencyStats()
        stats.record(100.0)
        d = stats.to_dict()
        assert d["p50_ms"] == 100.0
        assert d["p95_ms"] == 100.0
        assert d["sample_count"] == 1

    def test_reservoir_sliding_window(self) -> None:
        stats = LatencyStats()
        stats._max_samples = 3
        stats.record(1.0)
        stats.record(2.0)
        stats.record(3.0)
        stats.record(4.0)  # should evict 1.0
        assert stats.count == 3
        assert stats.samples == [2.0, 3.0, 4.0]


class TestProviderErrorStats:
    def test_empty(self) -> None:
        errors = ProviderErrorStats()
        d = errors.to_dict()
        assert d["by_provider"] == {}
        assert d["by_model"] == {}

    def test_record_errors(self) -> None:
        errors = ProviderErrorStats()
        errors.record("openai", "gpt-4")
        errors.record("openai", "gpt-4")
        errors.record("zai", "zai-model")
        d = errors.to_dict()
        assert d["by_provider"] == {"openai": 2, "zai": 1}
        assert d["by_model"] == {"gpt-4": 2, "zai-model": 1}


class TestTierDistribution:
    def test_empty(self) -> None:
        dist = TierDistribution()
        assert dist.to_dict() == {}

    def test_record_tiers(self) -> None:
        dist = TierDistribution()
        dist.record(1)
        dist.record(1)
        dist.record(3)
        assert dist.to_dict() == {"tier_1": 2, "tier_3": 1}


class TestStatsSnapshot:
    def test_empty_to_dict(self) -> None:
        snap = StatsSnapshot()
        d = snap.to_dict()
        assert d["total_requests"] == 0
        assert d["fallback"]["used"] == 0
        assert d["fallback"]["rate_pct"] == 0.0
        assert d["failed_requests"] == 0
        assert d["tier_distribution"] == {}
        assert d["latency"] == {}
        assert d["errors"] == {}
        assert d["cache"] == {}
        assert d["budget"] == {}

    def test_fallback_rate_calculation(self) -> None:
        snap = StatsSnapshot(total_requests=100, fallback_used=15)
        d = snap.to_dict()
        assert d["fallback"]["rate_pct"] == 15.0

    def test_fallback_rate_zero_division(self) -> None:
        snap = StatsSnapshot(total_requests=0, fallback_used=5)
        d = snap.to_dict()
        assert d["fallback"]["rate_pct"] == 0.0


class TestMetricsCollector:
    @pytest.mark.asyncio
    async def test_empty_snapshot(self) -> None:
        collector = MetricsCollector()
        snap = await collector.snapshot()
        assert snap.total_requests == 0
        assert snap.fallback_used == 0
        assert snap.failed_requests == 0

    @pytest.mark.asyncio
    async def test_record_request_basic(self) -> None:
        collector = MetricsCollector()
        await collector.record_request(tier=1, latency_ms=150.0)
        snap = await collector.snapshot()
        assert snap.total_requests == 1
        assert snap.tier_distribution == {"tier_1": 1}
        assert snap.latency["sample_count"] == 1
        assert snap.latency["p50_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_record_request_with_fallback(self) -> None:
        collector = MetricsCollector()
        await collector.record_request(
            tier=2, latency_ms=200.0, fallback_available=True, fallback_used=True
        )
        snap = await collector.snapshot()
        assert snap.total_requests == 1
        assert snap.fallback_available == 1
        assert snap.fallback_used == 1
        assert snap.tier_distribution == {"tier_2": 1}

    @pytest.mark.asyncio
    async def test_record_request_failed(self) -> None:
        collector = MetricsCollector()
        await collector.record_request(tier=1, latency_ms=50.0, failed=True)
        snap = await collector.snapshot()
        assert snap.total_requests == 1
        assert snap.failed_requests == 1

    @pytest.mark.asyncio
    async def test_record_request_stream(self) -> None:
        collector = MetricsCollector()
        await collector.record_request(
            tier=3, latency_ms=300.0, stream=True, fallback_used=True
        )
        snap = await collector.snapshot()
        assert snap.stream_requests == 1
        assert snap.stream_fallback_used == 1

    @pytest.mark.asyncio
    async def test_record_error(self) -> None:
        collector = MetricsCollector()
        await collector.record_error(provider="openai", model="gpt-4")
        await collector.record_error(provider="openai", model="gpt-4")
        await collector.record_error(provider="zai", model="zai-model")
        snap = await collector.snapshot()
        assert snap.errors["by_provider"] == {"openai": 2, "zai": 1}
        assert snap.errors["by_model"] == {"gpt-4": 2, "zai-model": 1}

    @pytest.mark.asyncio
    async def test_multiple_requests_aggregate(self) -> None:
        collector = MetricsCollector()
        for i in range(5):
            await collector.record_request(tier=1, latency_ms=100.0)
        for i in range(3):
            await collector.record_request(tier=2, latency_ms=200.0, fallback_available=True)
        await collector.record_request(tier=2, latency_ms=250.0, fallback_used=True)
        await collector.record_request(tier=3, latency_ms=50.0, failed=True)
        snap = await collector.snapshot()
        assert snap.total_requests == 10
        assert snap.fallback_available == 3
        assert snap.fallback_used == 1
        assert snap.failed_requests == 1
        assert snap.tier_distribution == {"tier_1": 5, "tier_2": 4, "tier_3": 1}
        assert snap.latency["sample_count"] == 10

    @pytest.mark.asyncio
    async def test_uptime_seconds(self) -> None:
        import time

        collector = MetricsCollector()
        # uptime should be very small right after creation
        assert collector.uptime_seconds >= 0
        assert collector.uptime_seconds < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_records(self) -> None:
        import asyncio

        collector = MetricsCollector()

        async def record_batch(n: int) -> None:
            for _ in range(n):
                await collector.record_request(tier=1, latency_ms=10.0)

        await asyncio.gather(record_batch(50), record_batch(50))
        snap = await collector.snapshot()
        assert snap.total_requests == 100
        assert snap.latency["sample_count"] == 100