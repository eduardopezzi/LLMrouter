"""Provider proxy with fallback handling."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from llmrouter.core.cooldown import ProviderCooldownStore
from llmrouter.core.health import ModelHealthTracker
from llmrouter.core.types import ChatRequest, ChatResponse, Provider, RoutingDecision
from llmrouter.logging_config import get_logger
from llmrouter.providers.base import BaseProvider, ProviderError

_logger = get_logger("llmrouter.proxy")


class ProviderProxy:
    """Dispatch routed requests to providers, trying fallbacks on retryable errors."""

    def __init__(
        self,
        providers: dict[Provider, BaseProvider],
        *,
        on_provider_error: Callable[[Any, ProviderError], None] | None = None,
        health_tracker: ModelHealthTracker | None = None,
        provider_cooldowns: ProviderCooldownStore | None = None,
    ) -> None:
        self._providers = providers
        self._on_provider_error = on_provider_error
        self._health_tracker = health_tracker
        self._disabled_providers: set[Provider] = set()
        self._provider_cooldowns = provider_cooldowns

    @property
    def providers(self) -> frozenset[Provider]:
        """Configured provider identifiers."""
        return frozenset(
            provider
            for provider in self._providers
            if provider not in self._disabled_providers
            and (
                self._provider_cooldowns is None
                or self._provider_cooldowns.is_provider_available(provider)
            )
        )

    def disable_provider(self, provider: Provider) -> None:
        """Stop sending new attempts to a provider for this process lifetime."""
        self._disabled_providers.add(provider)

    def set_provider_cooldowns(self, cooldowns: ProviderCooldownStore | None) -> None:
        """Inject provider/model cooldown memory."""
        self._provider_cooldowns = cooldowns

    async def chat_completion(
        self,
        request: ChatRequest,
        decision: RoutingDecision,
    ) -> ChatResponse:
        """Call the primary model and configured fallbacks."""
        attempts = _unique_attempts([decision.primary, *decision.fallbacks])
        last_error: ProviderError | None = None

        for i, model in enumerate(attempts):
            provider = self._providers.get(model.provider)
            cooldown = self._cooldown_for_model(model)
            if cooldown is not None:
                _logger.warning(
                    "Provider '%s' in cooldown for model '%s' for %.0fs",
                    model.provider.value,
                    model.name,
                    cooldown.seconds_remaining,
                )
                last_error = ProviderError(
                    f"Provider {model.provider.value} is in quota cooldown",
                    status_code=429,
                    provider=model.provider.value,
                )
                continue
            if model.provider in self._disabled_providers:
                _logger.warning(
                    "Provider '%s' disabled for model '%s'",
                    model.provider.value,
                    model.name,
                )
                last_error = ProviderError(
                    f"Provider {model.provider.value} is disabled",
                    status_code=503,
                    provider=model.provider.value,
                )
                continue
            if provider is None:
                _logger.warning(
                    "Provider '%s' not configured for model '%s'",
                    model.provider.value,
                    model.name,
                )
                last_error = ProviderError(
                    f"Provider {model.provider.value} is not configured",
                    status_code=503,
                    provider=model.provider.value,
                )
                continue

            try:
                _logger.debug(
                    "Trying provider '%s' (%s) [%d/%d]",
                    model.provider.value,
                    model.provider_model_name,
                    i + 1,
                    len(attempts),
                )
                response = await provider.chat_completion(request, model.provider_model_name)
                await self._record_success(model, response)
                return response
            except ProviderError as exc:
                await self._record_error(model, exc)
                self._record_cooldown(model, exc)
                self._handle_provider_error(model, exc)
                fallback_message = (
                    f" → falling back to '{attempts[i + 1].name}'"
                    if i + 1 < len(attempts)
                    else " (no more fallbacks)"
                )
                _logger.warning(
                    "Provider '%s' failed: %s (status=%d)%s",
                    model.provider.value,
                    exc,
                    exc.status_code,
                    fallback_message,
                )
                last_error = exc

        if last_error is not None:
            raise last_error
        raise ProviderError("No provider attempts were available", status_code=503)

    async def stream_chat_completion(
        self,
        request: ChatRequest,
        decision: RoutingDecision,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the primary model response, trying fallbacks on retryable errors.

        Yields parsed SSE chunk dictionaries in OpenAI format. On retryable errors
        from the primary model, falls back to the next model in the decision chain.
        """
        attempts = _unique_attempts([decision.primary, *decision.fallbacks])
        last_error: ProviderError | None = None

        for i, model in enumerate(attempts):
            provider = self._providers.get(model.provider)
            cooldown = self._cooldown_for_model(model)
            if cooldown is not None:
                last_error = ProviderError(
                    f"Provider {model.provider.value} is in quota cooldown",
                    status_code=429,
                    provider=model.provider.value,
                )
                continue
            if model.provider in self._disabled_providers:
                last_error = ProviderError(
                    f"Provider {model.provider.value} is disabled",
                    status_code=503,
                    provider=model.provider.value,
                )
                continue
            if provider is None:
                last_error = ProviderError(
                    f"Provider {model.provider.value} is not configured",
                    status_code=503,
                    provider=model.provider.value,
                )
                continue

            try:
                # Check if provider supports streaming
                if not hasattr(provider, "stream_completion"):
                    last_error = ProviderError(
                        f"Provider {model.provider.value} does not support streaming",
                        status_code=501,
                        provider=model.provider.value,
                    )
                    continue

                _logger.debug(
                    "Stream trying provider '%s' (%s) [%d/%d]",
                    model.provider.value,
                    model.provider_model_name,
                    i + 1,
                    len(attempts),
                )
                started = time.perf_counter()
                async for chunk in provider.stream_completion(request, model.provider_model_name):
                    yield chunk
                elapsed_ms = (time.perf_counter() - started) * 1000
                await self._record_stream_success(model, elapsed_ms)
                return  # Success — stop trying fallbacks
            except ProviderError as exc:
                await self._record_error(model, exc)
                self._record_cooldown(model, exc)
                self._handle_provider_error(model, exc)
                fallback_message = (
                    f" → falling back to '{attempts[i + 1].name}'"
                    if i + 1 < len(attempts)
                    else " (no more fallbacks)"
                )
                _logger.warning(
                    "Stream provider '%s' failed: %s (status=%d)%s",
                    model.provider.value,
                    exc,
                    exc.status_code,
                    fallback_message,
                )
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise ProviderError("No provider attempts were available for streaming", status_code=503)

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self._providers.values():
            await provider.close()

    def _handle_provider_error(self, model: Any, exc: ProviderError) -> None:
        if self._on_provider_error is None:
            return
        try:
            self._on_provider_error(model, exc)
        except Exception as callback_exc:  # pragma: no cover - defensive logging
            _logger.warning(
                "Provider error callback failed for model '%s': %s",
                getattr(model, "name", "(unknown)"),
                callback_exc,
            )

    def _cooldown_for_model(self, model: Any) -> Any | None:
        if self._provider_cooldowns is None:
            return None
        try:
            if not self._provider_cooldowns.is_model_available(model):
                return (
                    self._provider_cooldowns.model_cooldown(model.name)
                    or self._provider_cooldowns.provider_cooldown(model.provider)
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning("Provider cooldown check failed for '%s': %s", model.name, exc)
        return None

    def _record_cooldown(self, model: Any, exc: ProviderError) -> None:
        if self._provider_cooldowns is None:
            return
        try:
            entry = self._provider_cooldowns.record_quota_error(model, exc)
            if entry is not None:
                _logger.warning(
                    "Provider '%s' put in quota cooldown for %.0fs after model '%s': %s",
                    model.provider.value,
                    entry.seconds_remaining,
                    model.name,
                    exc,
                )
        except Exception as cooldown_exc:  # pragma: no cover - defensive logging
            _logger.warning(
                "Provider cooldown recording failed for '%s': %s",
                model.name,
                cooldown_exc,
            )

    async def _record_success(self, model: Any, response: ChatResponse) -> None:
        if self._health_tracker is None:
            return
        try:
            cost_usd = self._estimate_cost(model, response.usage)
            await self._health_tracker.record_success(
                model_name=model.name,
                latency_ms=response.latency_ms,
                cost_usd=cost_usd,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning("Health success recording failed for '%s': %s", model.name, exc)

    async def _record_stream_success(self, model: Any, elapsed_ms: float) -> None:
        if self._health_tracker is None:
            return
        try:
            await self._health_tracker.record_success(
                model_name=model.name,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning("Health stream success recording failed for '%s': %s", model.name, exc)

    async def _record_error(self, model: Any, exc: ProviderError) -> None:
        if self._health_tracker is None:
            return
        try:
            error_type = "timeout" if exc.status_code in {408, 504} else f"http_{exc.status_code}"
            await self._health_tracker.record_error(
                model_name=model.name,
                error_type=error_type,
            )
        except Exception as rec_exc:  # pragma: no cover - defensive logging
            _logger.warning("Health error recording failed for '%s': %s", model.name, rec_exc)

    @staticmethod
    def _estimate_cost(model: Any, usage: Any) -> float:
        input_cost = getattr(model, "cost_per_1k_input", 0.0) or 0.0
        output_cost = getattr(model, "cost_per_1k_output", 0.0) or 0.0
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return (prompt_tokens / 1000) * input_cost + (completion_tokens / 1000) * output_cost


def _unique_attempts(models: list[Any]) -> list[Any]:
    attempts: list[Any] = []
    seen: set[str] = set()
    for model in models:
        key = model.name
        if key in seen:
            continue
        attempts.append(model)
        seen.add(key)
    return attempts
