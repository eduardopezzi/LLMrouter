"""Provider proxy with fallback handling."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

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
    ) -> None:
        self._providers = providers
        self._on_provider_error = on_provider_error
        self._disabled_providers: set[Provider] = set()

    @property
    def providers(self) -> frozenset[Provider]:
        """Configured provider identifiers."""
        return frozenset(
            provider
            for provider in self._providers
            if provider not in self._disabled_providers
        )

    def disable_provider(self, provider: Provider) -> None:
        """Stop sending new attempts to a provider for this process lifetime."""
        self._disabled_providers.add(provider)

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
                return await provider.chat_completion(request, model.provider_model_name)
            except ProviderError as exc:
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
                async for chunk in provider.stream_completion(request, model.provider_model_name):
                    yield chunk
                return  # Success — stop trying fallbacks
            except ProviderError as exc:
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
