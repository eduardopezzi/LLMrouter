"""Operational metrics collector for LLMrouter.

Provides a thread-safe ``MetricsCollector`` that aggregates routing and
provider statistics across requests.  Designed to be injected into
``ProviderProxy`` and the API layer so that ``GET /v1/llmrouter/stats``
exposes a consolidated view without requiring manual log inspection.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """Rolling latency percentiles computed via reservoir sampling."""

    samples: list[float] = field(default_factory=list)
    _max_samples: int = 1000

    def record(self, latency_ms: float) -> None:
        if len(self.samples) >= self._max_samples:
            # Simple reservoir replacement: keep a sliding window
            self.samples.pop(0)
        self.samples.append(latency_ms)

    @property
    def p50(self) -> float:
        return _percentile(self.samples, 50)

    @property
    def p95(self) -> float:
        return _percentile(self.samples, 95)

    @property
    def count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "sample_count": self.count,
        }


@dataclass
class ProviderErrorStats:
    """Error counts grouped by provider and model."""

    by_provider: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, provider: str, model: str) -> None:
        self.by_provider[provider] += 1
        self.by_model[model] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_provider": dict(self.by_provider),
            "by_model": dict(self.by_model),
        }


@dataclass
class TierDistribution:
    """Count of requests routed to each tier."""

    counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, tier: int) -> None:
        self.counts[tier] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            f"tier_{tier}": count for tier, count in sorted(self.counts.items())
        }


@dataclass
class StatsSnapshot:
    """Immutable snapshot of all collected metrics."""

    total_requests: int = 0
    fallback_available: int = 0
    fallback_used: int = 0
    failed_requests: int = 0
    stream_requests: int = 0
    stream_fallback_used: int = 0
    tier_distribution: dict[str, int] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    # Placeholders for future phases
    cache: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "fallback": {
                "available": self.fallback_available,
                "used": self.fallback_used,
                "rate_pct": (
                    round(self.fallback_used / self.total_requests * 100, 2)
                    if self.total_requests > 0
                    else 0.0
                ),
            },
            "failed_requests": self.failed_requests,
            "stream": {
                "total": self.stream_requests,
                "fallback_used": self.stream_fallback_used,
            },
            "tier_distribution": self.tier_distribution,
            "latency": self.latency,
            "errors": self.errors,
            "cache": self.cache,
            "budget": self.budget,
        }


class MetricsCollector:
    """Thread-safe collector for operational metrics.

    Designed to be shared between the API layer and ``ProviderProxy``.
    All mutation methods are protected by an ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._total_requests: int = 0
        self._fallback_available: int = 0
        self._fallback_used: int = 0
        self._failed_requests: int = 0
        self._stream_requests: int = 0
        self._stream_fallback_used: int = 0
        self._tier_distribution = TierDistribution()
        self._latency = LatencyStats()
        self._errors = ProviderErrorStats()
        self._started_at: float = time.time()

    # ------------------------------------------------------------------
    # Recording helpers (called from routes / proxy)
    # ------------------------------------------------------------------

    async def record_request(
        self,
        *,
        tier: int,
        latency_ms: float,
        fallback_available: bool = False,
        fallback_used: bool = False,
        failed: bool = False,
        stream: bool = False,
    ) -> None:
        """Record a completed (or failed) request."""
        async with self._lock:
            self._total_requests += 1
            self._tier_distribution.record(tier)
            self._latency.record(latency_ms)
            if fallback_available:
                self._fallback_available += 1
            if fallback_used:
                self._fallback_used += 1
            if failed:
                self._failed_requests += 1
            if stream:
                self._stream_requests += 1
                if fallback_used:
                    self._stream_fallback_used += 1

    async def record_error(
        self,
        *,
        provider: str,
        model: str,
    ) -> None:
        """Record a provider error for stats aggregation."""
        async with self._lock:
            self._errors.record(provider, model)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    async def snapshot(self) -> StatsSnapshot:
        """Return an immutable snapshot of current metrics."""
        async with self._lock:
            return StatsSnapshot(
                total_requests=self._total_requests,
                fallback_available=self._fallback_available,
                fallback_used=self._fallback_used,
                failed_requests=self._failed_requests,
                stream_requests=self._stream_requests,
                stream_fallback_used=self._stream_fallback_used,
                tier_distribution=self._tier_distribution.to_dict(),
                latency=self._latency.to_dict(),
                errors=self._errors.to_dict(),
                cache={},
                budget={},
            )

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _percentile(sorted_or_unsorted: list[float], p: float) -> float:
    """Compute the *p*-th percentile (0-100) of a list of floats."""
    if not sorted_or_unsorted:
        return 0.0
    data = sorted(sorted_or_unsorted)
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    d = k - f
    return data[f] + d * (data[c] - data[f])