"""Runtime cooldown memory for providers and models.

This module keeps transient provider/model availability state in-process. It is
used for quota windows such as "usage limit reached; reset at ...", avoiding
repeated calls to a provider until the reset time passes.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from llmrouter.core.types import ModelInfo, Provider
from llmrouter.providers.base import ProviderError

UTC = timezone.utc  # noqa: UP017 - keep Python 3.10 compatibility.
_RESET_DATETIME = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T]"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:Z|[+-][0-9]{2}:?[0-9]{2})?"
)


@dataclass(frozen=True)
class CooldownEntry:
    """A provider/model cooldown entry."""

    provider: Provider
    model_name: str | None
    until: float
    reason: str

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.until - time.time())


class ProviderCooldownStore:
    """Thread-safe in-memory cooldown store."""

    def __init__(self, *, default_seconds: float = 5 * 60 * 60) -> None:
        self._default_seconds = default_seconds
        self._provider_until: dict[Provider, CooldownEntry] = {}
        self._model_until: dict[str, CooldownEntry] = {}
        self._lock = threading.RLock()

    def is_provider_available(self, provider: Provider, *, now: float | None = None) -> bool:
        """Return whether a provider is currently outside cooldown."""
        return self.provider_cooldown(provider, now=now) is None

    def is_model_available(self, model: ModelInfo, *, now: float | None = None) -> bool:
        """Return whether both provider and model are outside cooldown."""
        return (
            self.provider_cooldown(model.provider, now=now) is None
            and self.model_cooldown(model.name, now=now) is None
        )

    def provider_cooldown(
        self,
        provider: Provider,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Return an active provider cooldown entry, if present."""
        now = time.time() if now is None else now
        with self._lock:
            entry = self._provider_until.get(provider)
            if entry is None:
                return None
            if entry.until <= now:
                self._provider_until.pop(provider, None)
                return None
            return entry

    def model_cooldown(
        self,
        model_name: str,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Return an active model cooldown entry, if present."""
        now = time.time() if now is None else now
        with self._lock:
            entry = self._model_until.get(model_name)
            if entry is None:
                return None
            if entry.until <= now:
                self._model_until.pop(model_name, None)
                return None
            return entry

    def put_provider(
        self,
        provider: Provider,
        *,
        until: float,
        reason: str,
    ) -> CooldownEntry:
        """Put a provider in cooldown until a unix timestamp."""
        entry = CooldownEntry(provider=provider, model_name=None, until=until, reason=reason)
        with self._lock:
            current = self._provider_until.get(provider)
            if current is None or current.until < until:
                self._provider_until[provider] = entry
            return self._provider_until[provider]

    def put_model(
        self,
        model: ModelInfo,
        *,
        until: float,
        reason: str,
    ) -> CooldownEntry:
        """Put a specific model in cooldown until a unix timestamp."""
        entry = CooldownEntry(
            provider=model.provider,
            model_name=model.name,
            until=until,
            reason=reason,
        )
        with self._lock:
            current = self._model_until.get(model.name)
            if current is None or current.until < until:
                self._model_until[model.name] = entry
            return self._model_until[model.name]

    def record_quota_error(self, model: ModelInfo, exc: ProviderError) -> CooldownEntry | None:
        """Record provider cooldown when an error looks like quota exhaustion."""
        if not is_quota_exhaustion_error(exc):
            return None
        reset_at = quota_reset_timestamp(str(exc), default_seconds=self._default_seconds)
        return self.put_provider(model.provider, until=reset_at, reason=str(exc)[:300])

    def active_entries(self, *, now: float | None = None) -> list[CooldownEntry]:
        """Return active provider/model cooldown entries and prune expired ones."""
        now = time.time() if now is None else now
        entries: list[CooldownEntry] = []
        with self._lock:
            for provider in list(self._provider_until):
                entry = self.provider_cooldown(provider, now=now)
                if entry is not None:
                    entries.append(entry)
            for model_name in list(self._model_until):
                entry = self.model_cooldown(model_name, now=now)
                if entry is not None:
                    entries.append(entry)
        return entries


def is_quota_exhaustion_error(exc: ProviderError) -> bool:
    """Return True for provider quota/balance/rate-limit exhaustion."""
    if exc.status_code not in {402, 429}:
        return False
    message = str(exc).lower()
    indicators = (
        "usage limit reached",
        "rate limit",
        "quota",
        "quota exceeded",
        "insufficient balance",
        "insufficient quota",
        "insufficient credits",
        "no available resource",
        "billing",
        "credit",
        "credits",
        "recharge",
        "余额不足",
        "无可用资源包",
        "请充值",
    )
    return any(indicator in message for indicator in indicators)


def quota_reset_timestamp(message: str, *, default_seconds: float) -> float:
    """Infer quota reset unix timestamp from provider message."""
    parsed = _parse_reset_datetime(message)
    if parsed is not None:
        return max(time.time(), parsed.timestamp())

    duration = _parse_duration_seconds(message)
    if duration is not None:
        return time.time() + duration

    return time.time() + default_seconds


def _parse_reset_datetime(message: str) -> datetime | None:
    patterns = (
        rf"reset(?:s|ting)?\s+at\s+({_RESET_DATETIME})",
        rf"try\s+again\s+at\s+({_RESET_DATETIME})",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).replace(" ", "T")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        if re.search(r"[+-][0-9]{4}$", raw):
            raw = raw[:-2] + ":" + raw[-2:]
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    try:
        parsed_email = parsedate_to_datetime(message)
    except (TypeError, ValueError):
        return None
    if parsed_email.tzinfo is None:
        parsed_email = parsed_email.replace(tzinfo=UTC)
    return parsed_email.astimezone(UTC)


def _parse_duration_seconds(message: str) -> float | None:
    match = re.search(
        r"(?:for|in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*(second|seconds|minute|minutes|hour|hours|day|days)",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = 1
    if unit.startswith("minute"):
        multiplier = 60
    elif unit.startswith("hour"):
        multiplier = 60 * 60
    elif unit.startswith("day"):
        multiplier = 24 * 60 * 60
    return value * multiplier
