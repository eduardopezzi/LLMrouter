"""Provider proxy with fallback handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llmrouter.core.types import ChatRequest, ChatResponse, Provider, RoutingDecision
from llmrouter.providers.base import BaseProvider, ProviderError


class ProviderProxy:
    """Dispatch routed requests to providers, trying fallbacks on retryable errors."""

    def __init__(self, providers: dict[Provider, BaseProvider]) -> None:
        self._providers = providers

    @property
    def providers(self) -> frozenset[Provider]:
        """Configured provider identifiers."""
        return frozenset(self._providers)

    async def chat_completion(
        self,
        request: ChatRequest,
        decision: RoutingDecision,
    ) -> ChatResponse:
        """Call the primary model and configured fallbacks."""
        attempts = [decision.primary, *decision.fallbacks]
        last_error: ProviderError | None = None

        for model in attempts:
            provider = self._providers.get(model.provider)
            if provider is None:
                last_error = ProviderError(
                    f"Provider {model.provider.value} is not configured",
                    status_code=503,
                    provider=model.provider.value,
                )
                continue

            try:
                return await provider.chat_completion(request, model.provider_model_name)
            except ProviderError as exc:
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
        attempts = [decision.primary, *decision.fallbacks]
        last_error: ProviderError | None = None

        for model in attempts:
            provider = self._providers.get(model.provider)
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

                async for chunk in provider.stream_completion(request, model.provider_model_name):
                    yield chunk
                return  # Success — stop trying fallbacks
            except ProviderError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise ProviderError("No provider attempts were available for streaming", status_code=503)

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self._providers.values():
            await provider.close()
