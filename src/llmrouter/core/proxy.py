"""Provider proxy with fallback handling."""

from __future__ import annotations

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

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self._providers.values():
            await provider.close()
