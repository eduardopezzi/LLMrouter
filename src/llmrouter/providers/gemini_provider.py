"""Gemini provider placeholder.

Gemini does not share the OpenAI chat-completions wire format, so this
provider is intentionally explicit until a full adapter is implemented.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from llmrouter.core.types import ChatRequest, ChatResponse
from llmrouter.providers.base import BaseProvider, ProviderError


class GeminiProvider(BaseProvider):
    """Provider shell for Google Gemini."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            name="gemini",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        raise ProviderError(
            "Gemini provider adapter is not implemented yet",
            status_code=501,
            provider=self.name,
        )

    async def stream_completion(
        self, request: ChatRequest, model: str
    ) -> AsyncIterator[dict[str, object]]:
        raise ProviderError(
            "Gemini streaming adapter is not implemented yet",
            status_code=501,
            provider=self.name,
        )
        yield {}
