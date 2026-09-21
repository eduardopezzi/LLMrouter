"""Budget management with SQLite backend for per-tenant cost governance.

Tracks LLM spend per tenant (``project_id``, ``user_id``) against optional
daily and monthly USD limits.  Modes:

- ``soft``: exceeding a limit emits a :class:`BudgetDecision` warning but the
  request stays allowed.
- ``hard``: exceeding a limit denies the request with a reason.

Usage periods reset automatically when the UTC day/month rolls over (rows are
keyed by period, so historical periods are retained and reads simply target
the current period).

Follows the storage pattern of
:class:`llmrouter.core.cache.SQLiteCacheBackend`: on-demand connections, an
``asyncio.Lock`` serializing writes, and ``CREATE TABLE IF NOT EXISTS``
bootstrap.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.budget")

DEFAULT_PROJECT_ID = "default"
DEFAULT_USER_ID = "default"

BudgetMode = Literal["soft", "hard"]


@dataclass(frozen=True)
class BudgetLimits:
    """Per-tenant budget configuration.

    Attributes:
        daily_limit_usd: Maximum USD spend per UTC day; ``None`` disables the
            daily check.
        monthly_limit_usd: Maximum USD spend per UTC month; ``None`` disables
            the monthly check.
        mode: ``soft`` only warns when exceeded; ``hard`` denies.
    """

    daily_limit_usd: float | None = None
    monthly_limit_usd: float | None = None
    mode: BudgetMode = "soft"


@dataclass(frozen=True)
class BudgetUsage:
    """Snapshot of a tenant's spend and configuration for the current period.

    Attributes:
        project_id: Tenant project identifier.
        user_id: Tenant user identifier.
        daily_spent_usd: Spend recorded for ``period_day``.
        monthly_spent_usd: Spend recorded for ``period_month``.
        period_day: Current UTC day as ``YYYY-MM-DD``.
        period_month: Current UTC month as ``YYYY-MM``.
        daily_limit_usd: Active daily limit (``None`` when unset).
        monthly_limit_usd: Active monthly limit (``None`` when unset).
        mode: Active enforcement mode.
    """

    project_id: str
    user_id: str
    daily_spent_usd: float
    monthly_spent_usd: float
    period_day: str
    period_month: str
    daily_limit_usd: float | None
    monthly_limit_usd: float | None
    mode: BudgetMode


@dataclass(frozen=True)
class BudgetDecision:
    """Pre-flight verdict for a request against a tenant's budget.

    Attributes:
        allowed: Whether the request may proceed.
        reason: Denial motive when ``allowed`` is ``False`` (hard mode only).
        warning: Non-blocking notice when a limit is exceeded in soft mode.
        remaining_usd: Estimated remaining headroom (minimum across the active
            limits) when at least one limit is configured; otherwise ``None``.
    """

    allowed: bool
    reason: str | None = None
    warning: str | None = None
    remaining_usd: float | None = None


def estimate_cost(
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost from per-1k prices and token counts.

    Mirrors ``proxy._estimate_cost`` (``proxy.py`` ~606-611): the formula is
    intentionally replicated here so the budget module stays decoupled from
    the proxy for future enforcement wiring (B2).

    Args:
        cost_per_1k_input: USD per 1,000 input tokens.
        cost_per_1k_output: USD per 1,000 output tokens.
        input_tokens: Prompt token count.
        output_tokens: Completion token count.

    Returns:
        Estimated cost in USD.
    """
    input_cost = cost_per_1k_input or 0.0
    output_cost = cost_per_1k_output or 0.0
    return (input_tokens / 1000) * input_cost + (output_tokens / 1000) * output_cost


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_tenant(project_id: str | None, user_id: str | None) -> tuple[str, str]:
    """Map missing tenant identifiers to the shared default tenant."""
    return (
        project_id if project_id else DEFAULT_PROJECT_ID,
        user_id if user_id else DEFAULT_USER_ID,
    )


class BudgetManager:
    """SQLite-backed per-tenant budget tracking and pre-flight checking.

    Connections are opened on demand per operation (same pattern as
    :class:`~llmrouter.core.cache.SQLiteCacheBackend`) and every write is
    serialized through an ``asyncio.Lock`` so concurrent ``record_usage``
    calls cannot lose increments within a single event loop.
    """

    def __init__(self, db_path: str, clock: Callable[[], datetime] | None = None) -> None:
        """Initialize the manager.

        Args:
            db_path: Path of the SQLite database file (created on first use).
            clock: Injectable clock for period-rollover tests; defaults to
                ``datetime.now(timezone.utc)``.  Must return aware datetimes.
        """
        self._db_path = Path(db_path)
        self._clock = clock or _utc_now
        self._lock = asyncio.Lock()
        self._zero_cost_warned: set[tuple[str, str, str, str]] = set()

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    async def _ensure_tables(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_limits (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    daily_limit_usd REAL,
                    monthly_limit_usd REAL,
                    mode TEXT NOT NULL DEFAULT 'soft',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, user_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_usage (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    period TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK (scope IN ('day', 'month')),
                    spent_usd REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (project_id, user_id, period, scope)
                )
            """)
            conn.commit()

    async def _read_limits(
        self, conn: sqlite3.Connection, project_id: str, user_id: str
    ) -> BudgetLimits | None:
        row = conn.execute(
            """SELECT daily_limit_usd, monthly_limit_usd, mode
               FROM budget_limits
               WHERE project_id = ? AND user_id = ?""",
            (project_id, user_id),
        ).fetchone()
        if row is None:
            return None
        daily, monthly, mode = row
        return BudgetLimits(
            daily_limit_usd=daily,
            monthly_limit_usd=monthly,
            mode=mode if mode in ("soft", "hard") else "soft",
        )

    async def _read_spent(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        user_id: str,
        day: str,
        month: str,
    ) -> tuple[float, float]:
        rows = conn.execute(
            """SELECT period, scope, spent_usd FROM budget_usage
               WHERE project_id = ? AND user_id = ?
                 AND ((scope = 'day' AND period = ?) OR
                      (scope = 'month' AND period = ?))""",
            (project_id, user_id, day, month),
        ).fetchall()
        daily_spent = 0.0
        monthly_spent = 0.0
        for _period, scope, spent in rows:
            if scope == "day":
                daily_spent = spent
            else:
                monthly_spent = spent
        return daily_spent, monthly_spent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def set_limits(self, project_id: str, user_id: str, limits: BudgetLimits) -> None:
        """Persist (or replace) the budget limits for a tenant."""
        project_id, user_id = _normalize_tenant(project_id, user_id)
        await self._ensure_tables()
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO budget_limits
                       (project_id, user_id, daily_limit_usd, monthly_limit_usd,
                        mode, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        project_id,
                        user_id,
                        limits.daily_limit_usd,
                        limits.monthly_limit_usd,
                        limits.mode,
                        self._clock().isoformat(),
                    ),
                )
                conn.commit()

    async def get_usage(self, project_id: str, user_id: str) -> BudgetUsage:
        """Return the tenant's current-period spend snapshot.

        No row is written by this call; tenants without recorded usage read
        as zero spend with whatever limits are persisted (if any).
        """
        project_id, user_id = _normalize_tenant(project_id, user_id)
        await self._ensure_tables()
        now = self._clock()
        day = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")
        with self._connect() as conn:
            limits = await self._read_limits(conn, project_id, user_id)
            daily_spent, monthly_spent = await self._read_spent(
                conn, project_id, user_id, day, month
            )
        if limits is None:
            limits = BudgetLimits()
        return BudgetUsage(
            project_id=project_id,
            user_id=user_id,
            daily_spent_usd=daily_spent,
            monthly_spent_usd=monthly_spent,
            period_day=day,
            period_month=month,
            daily_limit_usd=limits.daily_limit_usd,
            monthly_limit_usd=limits.monthly_limit_usd,
            mode=limits.mode,
        )

    async def check(
        self, project_id: str, user_id: str, estimated_cost_usd: float = 0.0
    ) -> BudgetDecision:
        """Pre-flight budget verdict without recording any spend.

        Args:
            project_id: Tenant project identifier.
            user_id: Tenant user identifier.
            estimated_cost_usd: Anticipated cost of the upcoming request.

        Returns:
            A :class:`BudgetDecision`.  Read-only: repeated calls are
            idempotent.  With no limits configured the decision is always
            ``allowed`` with no warning.
        """
        usage = await self.get_usage(project_id, user_id)
        projected_daily = usage.daily_spent_usd + estimated_cost_usd
        projected_monthly = usage.monthly_spent_usd + estimated_cost_usd

        violations: list[tuple[str, float]] = []
        if usage.daily_limit_usd is not None and projected_daily > usage.daily_limit_usd:
            violations.append(("daily budget exceeded", projected_daily - usage.daily_limit_usd))
        if (
            usage.monthly_limit_usd is not None
            and projected_monthly > usage.monthly_limit_usd
        ):
            violations.append(
                ("monthly budget exceeded", projected_monthly - usage.monthly_limit_usd)
            )

        remaining: float | None = None
        if usage.daily_limit_usd is not None or usage.monthly_limit_usd is not None:
            candidates = [
                limit - usage.daily_spent_usd
                for limit in (usage.daily_limit_usd,)
                if limit is not None
            ] + [
                limit - usage.monthly_spent_usd
                for limit in (usage.monthly_limit_usd,)
                if limit is not None
            ]
            remaining = min(candidates)

        if not violations:
            return BudgetDecision(allowed=True, remaining_usd=remaining)

        worst = max(violations, key=lambda item: item[1])
        if usage.mode == "hard":
            return BudgetDecision(allowed=False, reason=worst[0], remaining_usd=remaining)
        return BudgetDecision(allowed=True, warning=worst[0], remaining_usd=remaining)

    async def record_usage(
        self,
        project_id: str,
        user_id: str,
        cost_usd: float,
        *,
        day: date | None = None,
        month: str | None = None,
        cost_known: bool = True,
    ) -> None:
        """Add ``cost_usd`` to the tenant's daily and monthly spend.

        Spend accumulates per ``(project_id, user_id, period)`` row, so a
        period change (UTC day/month rollover) starts from zero without any
        explicit reset call.  Historical rows are kept for auditability.

        Args:
            project_id: Tenant project identifier.
            user_id: Tenant user identifier.
            cost_usd: Cost to record in USD.
            day: Override the UTC day being credited (defaults to the
                injected clock's current day).
            month: Override the UTC month (``YYYY-MM``) being credited
                (defaults to the injected clock's current month).
            cost_known: ``False`` when the cost is zero because the model has
                no price in the catalog.  Triggers a
                ``budget_zero_cost_usage`` warning once per period per
                tenant.
        """
        project_id, user_id = _normalize_tenant(project_id, user_id)
        await self._ensure_tables()
        now = self._clock()
        period_day = day.isoformat() if day is not None else now.strftime("%Y-%m-%d")
        period_month = month if month is not None else now.strftime("%Y-%m")

        async with self._lock:
            with self._connect() as conn:
                for period, scope in ((period_day, "day"), (period_month, "month")):
                    conn.execute(
                        """INSERT INTO budget_usage
                           (project_id, user_id, period, scope, spent_usd)
                           VALUES (?, ?, ?, ?, ?)
                           ON CONFLICT (project_id, user_id, period, scope)
                           DO UPDATE SET spent_usd = spent_usd + excluded.spent_usd""",
                        (project_id, user_id, period, scope, cost_usd),
                    )
                conn.commit()

        if cost_usd == 0 and not cost_known:
            await self._warn_zero_cost_once(project_id, user_id, period_day, period_month)

    async def _warn_zero_cost_once(
        self, project_id: str, user_id: str, period_day: str, period_month: str
    ) -> None:
        """Emit the zero-cost structured warning once per tenant per period."""
        key = (project_id, user_id, period_day, period_month)
        async with self._lock:
            already_warned = key in self._zero_cost_warned
            if not already_warned:
                self._zero_cost_warned.add(key)
        if not already_warned:
            _logger.warning(
                "event=budget_zero_cost_usage project_id=%s user_id=%s period_day=%s "
                "period_month=%s cost_usd=0.0 reason=model_price_missing",
                project_id,
                user_id,
                period_day,
                period_month,
            )
