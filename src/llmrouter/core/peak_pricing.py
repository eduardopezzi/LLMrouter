"""Time-aware provider priority rules for peak/off-peak pricing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from llmrouter.core.types import ModelInfo, Provider

UTC = timezone.utc  # noqa: UP017 - keep Python 3.10 compatibility.


@dataclass(frozen=True)
class ProviderPricingRule:
    """Describe when a provider charges peak prices in its billing timezone."""

    provider: Provider
    timezone_name: str
    off_peak_start: time
    off_peak_end: time
    weekend_off_peak_from: date | None = None

    def is_peak(self, instant: datetime) -> bool:
        """Return whether ``instant`` falls in this provider's peak period."""
        local = _as_aware_utc(instant).astimezone(ZoneInfo(self.timezone_name))
        if (
            self.weekend_off_peak_from is not None
            and local.date() >= self.weekend_off_peak_from
            and local.weekday() >= 5
        ):
            return False
        return not _time_in_window(
            local.time().replace(tzinfo=None),
            self.off_peak_start,
            self.off_peak_end,
        )


class PeakPricingPriorityPolicy:
    """Stably demote providers that are currently charging peak prices."""

    def __init__(
        self,
        rules: Iterable[ProviderPricingRule],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._clock = clock or (lambda: datetime.now(UTC))

    def peak_providers(self, *, instant: datetime | None = None) -> set[Provider]:
        """Return providers whose configured price period is currently peak."""
        current = instant or self._clock()
        return {rule.provider for rule in self._rules if rule.is_peak(current)}

    def prioritize(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Move peak-priced providers behind alternatives without removing them."""
        peak = self.peak_providers()
        if not peak:
            return models
        regular = [model for model in models if model.provider not in peak]
        expensive = [model for model in models if model.provider in peak]
        return [*regular, *expensive]


def _as_aware_utc(instant: datetime) -> datetime:
    """Treat naive datetimes as UTC and normalize aware values to UTC."""
    if instant.tzinfo is None:
        return instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC)


def _time_in_window(value: time, start: time, end: time) -> bool:
    """Return whether a local time belongs to a possibly overnight window."""
    if start == end:
        return True
    if start < end:
        return start <= value < end
    return value >= start or value < end
