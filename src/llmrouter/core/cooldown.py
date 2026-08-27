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
from enum import Enum

from llmrouter.core.types import ModelInfo, Provider
from llmrouter.providers.base import ProviderError

UTC = timezone.utc  # noqa: UP017 - keep Python 3.10 compatibility.
_RESET_DATETIME = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T]"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:Z|[+-][0-9]{2}:?[0-9]{2})?"
)


class CooldownScope(str, Enum):
    """Availability boundary affected by a provider error."""

    PROVIDER = "provider"
    CLOUD = "cloud"
    MODEL = "model"


@dataclass(frozen=True)
class CooldownEntry:
    """A provider/model cooldown entry."""

    provider: Provider
    model_name: str | None
    until: float
    reason: str
    scope: CooldownScope
    failures: int = 1
    permanent: bool = False

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.until - time.time())


class ProviderCooldownStore:
    """Thread-safe in-memory cooldown store."""

    def __init__(
        self,
        *,
        default_seconds: float = 10 * 60,
        probe_retry_seconds: float = 60 * 60,
    ) -> None:
        self._default_seconds = default_seconds
        self._probe_retry_seconds = probe_retry_seconds
        self._provider_until: dict[Provider, CooldownEntry] = {}
        self._cloud_until: dict[Provider, CooldownEntry] = {}
        self._model_until: dict[str, CooldownEntry] = {}
        self._probe_in_flight: dict[str, CooldownEntry] = {}
        self._lock = threading.RLock()

    def is_provider_available(self, provider: Provider, *, now: float | None = None) -> bool:
        """Return whether a provider is currently outside cooldown."""
        return self.provider_cooldown(provider, now=now) is None

    def is_model_available(self, model: ModelInfo, *, now: float | None = None) -> bool:
        """Return whether provider, cloud family, and model are all available."""
        return self.cooldown_for_model(model, now=now) is None

    def cooldown_for_model(
        self,
        model: ModelInfo,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Return the most specific cooldown blocking ``model``."""
        del now  # Expired entries remain blocked until a half-open probe succeeds.
        with self._lock:
            model_entry = self._model_until.get(model.name)
            if model_entry is not None:
                return model_entry
            if is_ollama_cloud_model(model):
                cloud_entry = self._cloud_until.get(model.provider)
                if cloud_entry is not None:
                    return cloud_entry
            return self._provider_until.get(model.provider)

    def provider_cooldown(
        self,
        provider: Provider,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Return an active provider cooldown entry, if present."""
        del now
        with self._lock:
            return self._provider_until.get(provider)

    def cloud_cooldown(
        self,
        provider: Provider,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Return a cloud-family cooldown, including a due half-open state."""
        del now
        with self._lock:
            return self._cloud_until.get(provider)

    def model_cooldown(
        self,
        model_name: str,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Return an active model cooldown entry, if present."""
        del now
        with self._lock:
            return self._model_until.get(model_name)

    def put_provider(
        self,
        provider: Provider,
        *,
        until: float,
        reason: str,
    ) -> CooldownEntry:
        """Put a provider in cooldown until a unix timestamp."""
        entry = CooldownEntry(
            provider=provider,
            model_name=None,
            until=until,
            reason=reason,
            scope=CooldownScope.PROVIDER,
        )
        with self._lock:
            current = self._provider_until.get(provider)
            if current is None or current.until < until:
                self._provider_until[provider] = entry
            return self._provider_until[provider]

    def put_cloud(
        self,
        provider: Provider,
        *,
        model_name: str,
        until: float,
        reason: str,
    ) -> CooldownEntry:
        """Put all cloud models reached through one provider in cooldown."""
        entry = CooldownEntry(
            provider=provider,
            model_name=model_name,
            until=until,
            reason=reason,
            scope=CooldownScope.CLOUD,
        )
        with self._lock:
            current = self._cloud_until.get(provider)
            if current is None or current.until < until:
                self._cloud_until[provider] = entry
            return self._cloud_until[provider]

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
            scope=CooldownScope.MODEL,
        )
        with self._lock:
            current = self._model_until.get(model.name)
            if current is None or current.until < until:
                self._model_until[model.name] = entry
            return self._model_until[model.name]

    def record_error(
        self,
        model: ModelInfo,
        exc: ProviderError,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Classify an upstream failure and record the appropriate cooldown scope."""
        current = time.time() if now is None else now
        reason = str(exc)[:300]
        if is_permanent_model_error(exc):
            entry = CooldownEntry(
                provider=model.provider,
                model_name=model.name,
                until=float("inf"),
                reason=reason,
                scope=CooldownScope.MODEL,
                permanent=True,
            )
            with self._lock:
                self._model_until[model.name] = entry
            return entry
        if is_model_unavailable_error(exc):
            return self.put_model(
                model,
                until=current + self._default_seconds,
                reason=reason,
            )
        if not is_quota_exhaustion_error(exc):
            return None
        until = current + self._default_seconds
        if model.provider == Provider.OLLAMA:
            if is_ollama_cloud_model(model):
                return self.put_cloud(
                    model.provider,
                    model_name=model.name,
                    until=until,
                    reason=reason,
                )
            return self.put_model(model, until=until, reason=reason)
        entry = CooldownEntry(
            provider=model.provider,
            model_name=model.name,
            until=until,
            reason=reason,
            scope=CooldownScope.PROVIDER,
        )
        with self._lock:
            current_entry = self._provider_until.get(model.provider)
            if current_entry is None or current_entry.until < until:
                self._provider_until[model.provider] = entry
            return self._provider_until[model.provider]

    def record_quota_error(
        self,
        model: ModelInfo,
        exc: ProviderError,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Backward-compatible quota-only entry point."""
        if not is_quota_exhaustion_error(exc):
            return None
        return self.record_error(model, exc, now=now)

    def claim_due_probes(
        self,
        models: list[ModelInfo],
        *,
        now: float | None = None,
    ) -> tuple[ModelInfo, ...]:
        """Atomically claim one representative model for every due cooldown."""
        current = time.time() if now is None else now
        by_name = {model.name: model for model in models}
        claimed: list[ModelInfo] = []
        with self._lock:
            entries = [
                *self._provider_until.values(),
                *self._cloud_until.values(),
                *self._model_until.values(),
            ]
            for entry in entries:
                if entry.permanent or entry.until > current:
                    continue
                model = self._probe_model(entry, models, by_name)
                if model is None or model.name in self._probe_in_flight:
                    continue
                self._probe_in_flight[model.name] = entry
                claimed.append(model)
        return tuple(claimed)

    def probe_succeeded(self, model: ModelInfo) -> CooldownEntry | None:
        """Close a half-open cooldown after a successful background canary."""
        with self._lock:
            entry = self._probe_in_flight.pop(model.name, None)
            if entry is not None:
                self._remove_entry(entry)
            return entry

    def probe_failed(
        self,
        model: ModelInfo,
        exc: ProviderError,
        *,
        now: float | None = None,
    ) -> CooldownEntry | None:
        """Reopen a failed half-open cooldown using the longer retry interval."""
        current = time.time() if now is None else now
        with self._lock:
            claimed = self._probe_in_flight.pop(model.name, None)
            if claimed is None:
                return None
            if is_permanent_model_error(exc):
                self._remove_entry(claimed)
                entry = CooldownEntry(
                    provider=model.provider,
                    model_name=model.name,
                    until=float("inf"),
                    reason=str(exc)[:300],
                    scope=CooldownScope.MODEL,
                    failures=claimed.failures + 1,
                    permanent=True,
                )
                self._set_entry(entry)
                return entry
            if is_model_unavailable_error(exc) and claimed.scope != CooldownScope.MODEL:
                self._remove_entry(claimed)
                entry = CooldownEntry(
                    provider=model.provider,
                    model_name=model.name,
                    until=current + self._probe_retry_seconds,
                    reason=str(exc)[:300],
                    scope=CooldownScope.MODEL,
                    failures=claimed.failures + 1,
                )
                self._set_entry(entry)
                return entry
            entry = CooldownEntry(
                provider=claimed.provider,
                model_name=claimed.model_name,
                until=current + self._probe_retry_seconds,
                reason=str(exc)[:300],
                scope=claimed.scope,
                failures=claimed.failures + 1,
            )
            self._set_entry(entry)
            return entry

    def is_model_retired(self, model_name: str) -> bool:
        """Return whether an upstream response permanently retired a model."""
        entry = self.model_cooldown(model_name)
        return bool(entry and entry.permanent)

    def active_entries(self, *, now: float | None = None) -> list[CooldownEntry]:
        """Return blocking cooldown entries, including those waiting for a probe."""
        del now
        with self._lock:
            return [
                *self._provider_until.values(),
                *self._cloud_until.values(),
                *self._model_until.values(),
            ]

    @staticmethod
    def _probe_model(
        entry: CooldownEntry,
        models: list[ModelInfo],
        by_name: dict[str, ModelInfo],
    ) -> ModelInfo | None:
        if entry.model_name and entry.model_name in by_name:
            return by_name[entry.model_name]
        candidates = [model for model in models if model.provider == entry.provider]
        if entry.scope == CooldownScope.CLOUD:
            candidates = [model for model in candidates if is_ollama_cloud_model(model)]
        return candidates[0] if candidates else None

    def _remove_entry(self, entry: CooldownEntry) -> None:
        if entry.scope == CooldownScope.PROVIDER:
            self._provider_until.pop(entry.provider, None)
        elif entry.scope == CooldownScope.CLOUD:
            self._cloud_until.pop(entry.provider, None)
        elif entry.model_name:
            self._model_until.pop(entry.model_name, None)

    def _set_entry(self, entry: CooldownEntry) -> None:
        if entry.scope == CooldownScope.PROVIDER:
            self._provider_until[entry.provider] = entry
        elif entry.scope == CooldownScope.CLOUD:
            self._cloud_until[entry.provider] = entry
        elif entry.model_name:
            self._model_until[entry.model_name] = entry


def is_ollama_cloud_model(model: ModelInfo) -> bool:
    """Return whether an Ollama catalog entry is remotely hosted."""
    return model.provider == Provider.OLLAMA and model.provider_model_name.endswith(":cloud")


def is_permanent_model_error(exc: ProviderError) -> bool:
    """Return whether the upstream says a model has permanently left its catalog."""
    message = str(exc).lower()
    return exc.status_code == 410 or any(
        indicator in message
        for indicator in (
            "was retired",
            "has been retired",
            "model retired",
            "no longer available",
            "permanently deprecated",
            "removed from the catalog",
            "removed from catalog",
            "model discontinued",
        )
    )


def is_model_unavailable_error(exc: ProviderError) -> bool:
    """Return whether only the requested model appears unavailable."""
    if is_permanent_model_error(exc):
        return True
    message = str(exc).lower()
    if exc.status_code != 404 or "model" not in message:
        return False
    return any(
        indicator in message
        for indicator in ("not found", "unknown", "does not exist", "not in catalog")
    )


def is_quota_exhaustion_error(exc: ProviderError) -> bool:
    """Return True for provider quota/balance/rate-limit exhaustion."""
    if exc.status_code == 402:
        return True
    if exc.status_code != 429:
        return False
    message = str(exc).lower()
    indicators = (
        "usage limit",
        "usage limit reached",
        "session limit",
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
        return parsed.timestamp()

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
