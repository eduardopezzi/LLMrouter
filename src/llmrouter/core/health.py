"""Model health tracking with realtime metrics and composite HealthScore.

The tracker collects per-model success/error events and exposes aggregated
metrics (latency percentiles, error rate, average quality, average cost) plus
a composite ``HealthScore`` that routing strategies can use to prefer healthier
models.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.health")


class HealthBackend(str, Enum):
    """Supported persistence backends for health metrics."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    REDIS = "redis"


@dataclass(frozen=True)
class ModelHealth:
    """Aggregated health metrics for a single model.

    All latency values are in milliseconds. ``window_count`` is the number of
    events inside the configured sliding window.
    """

    model_name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_latency_ms: float
    error_rate: float
    avg_quality: float
    avg_cost_usd: float
    request_count: int
    window_start_ts: float
    window_end_ts: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "avg_quality": round(self.avg_quality, 2),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "request_count": self.request_count,
            "window_start_ts": self.window_start_ts,
            "window_end_ts": self.window_end_ts,
        }


@dataclass(frozen=True)
class HealthScore:
    """Composite score (0.0–1.0) used for routing. 1.0 = best health."""

    model_name: str
    score: float
    latency_score: float
    error_score: float
    quality_score: float
    cost_score: float
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "score": round(self.score, 4),
            "latency_score": round(self.latency_score, 4),
            "error_score": round(self.error_score, 4),
            "quality_score": round(self.quality_score, 4),
            "cost_score": round(self.cost_score, 4),
            "request_count": self.request_count,
        }


@dataclass(frozen=True)
class HealthEvent:
    """Single success or error event captured by the tracker."""

    model_name: str
    ts: float
    success: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    quality: float = 0.0
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "ts": self.ts,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "quality": self.quality,
            "error_type": self.error_type,
        }


class HealthStore(Protocol):
    """Protocol for health storage backends."""

    async def record_success(
        self,
        model_name: str,
        latency_ms: float,
        cost_usd: float,
        quality: float,
    ) -> None:
        """Persist a successful request event."""
        ...

    async def record_error(self, model_name: str, error_type: str) -> None:
        """Persist a failed request event."""
        ...

    async def get_health(
        self,
        model_name: str,
        window_minutes: int,
        now_ts: float,
    ) -> ModelHealth:
        """Return aggregated health for a model inside the sliding window."""
        ...

    async def list_models(self, window_minutes: int, now_ts: float) -> list[str]:
        """Return all model names that have events in the window."""
        ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryHealthStore:
    """Thread-safe in-memory sliding window store (default / tests)."""

    def __init__(self, max_events_per_model: int = 10_000) -> None:
        self._events: dict[str, deque[HealthEvent]] = {}
        self._lock = asyncio.Lock()
        self._max_events = max_events_per_model

    async def record_success(
        self,
        model_name: str,
        latency_ms: float,
        cost_usd: float,
        quality: float,
    ) -> None:
        event = HealthEvent(
            model_name=model_name,
            ts=time.time(),
            success=True,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            quality=quality,
        )
        async with self._lock:
            self._events.setdefault(model_name, deque(maxlen=self._max_events)).append(event)

    async def record_error(self, model_name: str, error_type: str = "provider_error") -> None:
        event = HealthEvent(
            model_name=model_name,
            ts=time.time(),
            success=False,
            error_type=error_type,
        )
        async with self._lock:
            self._events.setdefault(model_name, deque(maxlen=self._max_events)).append(event)

    async def get_health(
        self,
        model_name: str,
        window_minutes: int,
        now_ts: float,
    ) -> ModelHealth:
        window_start = now_ts - (window_minutes * 60)
        async with self._lock:
            events = list(self._events.get(model_name, deque()))
        return _aggregate(events, model_name, window_start, now_ts)

    async def list_models(self, window_minutes: int, now_ts: float) -> list[str]:
        window_start = now_ts - (window_minutes * 60)
        result: list[str] = []
        async with self._lock:
            for model_name, events in self._events.items():
                if any(e.ts >= window_start for e in events):
                    result.append(model_name)
        return result


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SQLiteHealthStore:
    """Persistent sliding-window store backed by SQLite.

    Events are kept with a TTL; old rows are pruned lazily on reads and during
    ``record_*`` calls via a background periodic cleanup (to keep writes fast).
    """

    def __init__(self, db_path: str = "data/health.db", ttl_minutes: int = 60) -> None:
        self._db_path = db_path
        self._ttl_minutes = ttl_minutes
        self._lock = asyncio.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self) -> None:
        with sqlite3.connect(self._db_path) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS health_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    ts REAL NOT NULL,
                    success INTEGER NOT NULL DEFAULT 1,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    quality REAL NOT NULL DEFAULT 0,
                    error_type TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_health_model_ts
                    ON health_events(model_name, ts);
                """
            )
            db.commit()

    async def record_success(
        self,
        model_name: str,
        latency_ms: float,
        cost_usd: float,
        quality: float,
    ) -> None:
        async with self._lock:
            with sqlite3.connect(self._db_path) as db:
                db.execute(
                    """
                    INSERT INTO health_events
                        (model_name, ts, success, latency_ms, cost_usd, quality, error_type)
                    VALUES (?, ?, 1, ?, ?, ?, '')
                    """,
                    (model_name, time.time(), latency_ms, cost_usd, quality),
                )
                db.commit()

    async def record_error(self, model_name: str, error_type: str = "provider_error") -> None:
        async with self._lock:
            with sqlite3.connect(self._db_path) as db:
                db.execute(
                    """
                    INSERT INTO health_events
                        (model_name, ts, success, latency_ms, cost_usd, quality, error_type)
                    VALUES (?, ?, 0, 0, 0, 0, ?)
                    """,
                    (model_name, time.time(), error_type),
                )
                db.commit()

    async def get_health(
        self,
        model_name: str,
        window_minutes: int,
        now_ts: float,
    ) -> ModelHealth:
        window_start = now_ts - (window_minutes * 60)
        cutoff = now_ts - (self._ttl_minutes * 60)
        events: list[HealthEvent] = []
        async with self._lock:
            with sqlite3.connect(self._db_path) as db:
                db.execute("DELETE FROM health_events WHERE ts < ?", (cutoff,))
                db.commit()
                rows = db.execute(
                    """
                    SELECT model_name, ts, success, latency_ms, cost_usd, quality, error_type
                    FROM health_events
                    WHERE model_name = ? AND ts >= ?
                    ORDER BY ts ASC
                    """,
                    (model_name, window_start),
                ).fetchall()
        events = [_row_to_event(row) for row in rows]
        return _aggregate(events, model_name, window_start, now_ts)

    async def list_models(self, window_minutes: int, now_ts: float) -> list[str]:
        window_start = now_ts - (window_minutes * 60)
        cutoff = now_ts - (self._ttl_minutes * 60)
        async with self._lock:
            with sqlite3.connect(self._db_path) as db:
                db.execute("DELETE FROM health_events WHERE ts < ?", (cutoff,))
                db.commit()
                rows = db.execute(
                    "SELECT DISTINCT model_name FROM health_events WHERE ts >= ?",
                    (window_start,),
                ).fetchall()
        return sorted({str(row[0]) for row in rows})


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate(events: list[HealthEvent], model_name: str, window_start: float, now_ts: float) -> ModelHealth:
    """Build :class:`ModelHealth` from a list of ``HealthEvent``."""
    in_window = [e for e in events if window_start <= e.ts <= now_ts]
    if not in_window:
        return ModelHealth(
            model_name=model_name,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            avg_latency_ms=0.0,
            error_rate=0.0,
            avg_quality=0.0,
            avg_cost_usd=0.0,
            request_count=0,
            window_start_ts=window_start,
            window_end_ts=now_ts,
        )

    success_events = [e for e in in_window if e.success]
    error_count = len(in_window) - len(success_events)
    error_rate = error_count / len(in_window)

    latencies = sorted(e.latency_ms for e in success_events if e.latency_ms > 0)
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    costs = [e.cost_usd for e in success_events]
    avg_cost = sum(costs) / len(costs) if costs else 0.0

    qualities = [e.quality for e in success_events if e.quality > 0]
    avg_quality = sum(qualities) / len(qualities) if qualities else 0.0

    return ModelHealth(
        model_name=model_name,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        avg_latency_ms=avg_latency,
        error_rate=error_rate,
        avg_quality=avg_quality,
        avg_cost_usd=avg_cost,
        request_count=len(in_window),
        window_start_ts=window_start,
        window_end_ts=now_ts,
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    """Return percentile ``q`` for an already sorted list of values.

    Uses linear interpolation to match NumPy semantics, independent of the
    ``statistics.quantiles`` method name availability across Python versions.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    index = q * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    fraction = index - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _row_to_event(row: tuple[Any, ...]) -> HealthEvent:
    return HealthEvent(
        model_name=str(row[0]),
        ts=float(row[1]),
        success=bool(row[2]),
        latency_ms=float(row[3]),
        cost_usd=float(row[4]),
        quality=float(row[5]),
        error_type=str(row[6]),
    )


# ---------------------------------------------------------------------------
# Tracker frontend
# ---------------------------------------------------------------------------


@dataclass
class HealthWeights:
    """Weights for the composite HealthScore. Should sum to ~1.0."""

    latency: float = 0.30
    error: float = 0.35
    quality: float = 0.25
    cost: float = 0.10


class ModelHealthTracker:
    """Realtime per-model health tracker with composite score.

    The tracker is safe to share across async tasks. It accepts raw success and
    error events and exposes both aggregated :class:`ModelHealth` snapshots and
    :class:`HealthScore` values suitable for adaptive routing.

    When ``log_health_summary`` is True, a summary of all models' HealthScore
    is logged at INFO level every ``log_interval_seconds`` (or every request
    if none set).
    """

    def __init__(
        self,
        store: HealthStore | None = None,
        *,
        window_minutes: int = 15,
        weights: HealthWeights | None = None,
        quality_source: Any | None = None,
        log_health_summary: bool = True,
        log_interval_seconds: int = 60,
    ) -> None:
        self._store = store or InMemoryHealthStore()
        self._window_minutes = window_minutes
        self._weights = weights or HealthWeights()
        self._quality_source = quality_source
        self._log_health_summary = log_health_summary
        self._log_interval = log_interval_seconds
        self._last_summary_log: float = 0.0
        self._request_count_total: int = 0
        self._request_count_since_log: int = 0

    @property
    def store(self) -> HealthStore:
        return self._store

    @property
    def window_minutes(self) -> int:
        return self._window_minutes

    async def record_success(
        self,
        model_name: str,
        latency_ms: float,
        cost_usd: float,
        quality: float | None = None,
    ) -> None:
        """Record a successful model invocation."""
        quality_value = quality if quality is not None else await self._lookup_quality(model_name)
        await self._store.record_success(model_name, latency_ms, cost_usd, quality_value)
        self._request_count_total += 1
        self._request_count_since_log += 1
        await self._maybe_log_summary()

    async def record_error(
        self,
        model_name: str,
        error_type: str = "provider_error",
    ) -> None:
        """Record a failed model invocation."""
        await self._store.record_error(model_name, error_type)
        self._request_count_total += 1
        self._request_count_since_log += 1
        await self._maybe_log_summary()

    async def _maybe_log_summary(self) -> None:
        """Log aggregated health summary at configured interval."""
        if not self._log_health_summary:
            return
        now = time.time()
        if now - self._last_summary_log < self._log_interval:
            return
        self._last_summary_log = now
        self._request_count_since_log = 0
        try:
            all_health = await self.list_health()
            all_scores = await self.score_map()
            if not all_health:
                return

            # Build summary per-model
            model_summaries: list[str] = []
            total_reqs = 0
            total_errors = 0
            for h in all_health:
                total_reqs += h.request_count
                total_errors += int(h.error_rate * h.request_count) if h.request_count > 0 else 0
                score = all_scores.get(h.model_name)
                score_str = f"{score.score:.4f}" if score else "n/a"
                p95_str = f"{h.p95_ms:.0f}" if h.p95_ms > 0 else "?"
                model_summaries.append(
                    f"{h.model_name}=score={score_str} p95={p95_str}ms "
                    f"err={h.error_rate:.2%} req={h.request_count}"
                )

            overall_error_rate = total_errors / total_reqs if total_reqs > 0 else 0.0
            _logger.info(
                "HealthSummary total_reqs=%d total_errors=%d error_rate=%.2f%% "
                "models=%s",
                total_reqs,
                total_errors,
                overall_error_rate * 100.0,
                " | ".join(model_summaries),
            )
        except Exception:
            _logger.debug("Health summary log failed (non-fatal)", exc_info=True)

    async def get_health(self, model_name: str) -> ModelHealth:
        """Return aggregated metrics for ``model_name`` in the current window."""
        now = time.time()
        return await self._store.get_health(model_name, self._window_minutes, now)

    async def list_health(self) -> list[ModelHealth]:
        """Return health snapshots for every model seen in the window."""
        now = time.time()
        model_names = await self._store.list_models(self._window_minutes, now)
        return [await self._store.get_health(name, self._window_minutes, now) for name in model_names]

    async def health_score(self, model_name: str) -> HealthScore:
        """Compute composite health score for a model (0.0–1.0, higher is better)."""
        health = await self.get_health(model_name)
        latency_score = _score_latency(health.p95_ms)
        error_score = _score_error_rate(health.error_rate)
        quality_score = _score_quality(health.avg_quality)
        cost_score = _score_cost(health.avg_cost_usd)
        composite = (
            latency_score * self._weights.latency
            + error_score * self._weights.error
            + quality_score * self._weights.quality
            + cost_score * self._weights.cost
        )
        return HealthScore(
            model_name=model_name,
            score=min(max(composite, 0.0), 1.0),
            latency_score=latency_score,
            error_score=error_score,
            quality_score=quality_score,
            cost_score=cost_score,
            request_count=health.request_count,
        )

    async def score_map(self) -> dict[str, HealthScore]:
        """Return a mapping of model name -> HealthScore for models in window."""
        now = time.time()
        names = await self._store.list_models(self._window_minutes, now)
        return {name: await self.health_score(name) for name in names}

    async def _lookup_quality(self, model_name: str) -> float:
        """Optional hook to load recent average quality from an external source."""
        if self._quality_source is None:
            return 0.0
        try:
            quality = await self._quality_source.get_average_quality(model_name)
            return float(quality) if quality is not None else 0.0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_latency(p95_ms: float) -> float:
    """Latency score: faster is better. Assumes 500ms ideal, 15000ms terrible."""
    if p95_ms <= 0:
        return 1.0
    return max(0.0, 1.0 - (p95_ms / 15000.0))


def _score_error_rate(error_rate: float) -> float:
    """Error score: zero errors = 1.0; 100% errors = 0.0 (exponential decay)."""
    return max(0.0, 1.0 - (error_rate**0.5))


def _score_quality(avg_quality: float) -> float:
    """Quality score: 1-5 scale mapped to 0-1."""
    if avg_quality <= 0:
        return 0.5  # neutral when unknown
    return min(max((avg_quality - 1.0) / 4.0, 0.0), 1.0)


def _score_cost(avg_cost_usd: float) -> float:
    """Cost score: cheaper is better. Assumes $0.10 avg is ideal baseline."""
    if avg_cost_usd <= 0:
        return 1.0
    return max(0.0, 1.0 - (avg_cost_usd / 0.10))


# ---------------------------------------------------------------------------
# Quality source helper for evaluator reviews
# ---------------------------------------------------------------------------


@dataclass
class ReviewQualitySource:
    """Fetch average review quality per model from the evaluator SQLite DB."""

    db_path: str = "data/llmrouter.db"
    window_minutes: int = 60

    async def get_average_quality(self, model_name: str) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._query, model_name)

    def _query(self, model_name: str) -> float:
        path = Path(self.db_path)
        if not path.exists():
            return 0.0
        with sqlite3.connect(self.db_path) as db:
            if not _table_exists(db, "reviews") or not _table_exists(db, "observations"):
                return 0.0
            cutoff = (datetime.now(timezone.utc).timestamp() - self.window_minutes * 60) or 0
            row = db.execute(
                """
                SELECT COALESCE(AVG(r.quality_overall), 0)
                FROM reviews r
                JOIN observations o ON o.id = r.observation_id
                WHERE o.chosen_model = ? AND o.created_at >= datetime(?, 'unixepoch')
                """,
                (model_name, cutoff),
            ).fetchone()
        return float(row[0]) if row else 0.0


def _table_exists(db: sqlite3.Connection, table_name: str) -> bool:
    row = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None