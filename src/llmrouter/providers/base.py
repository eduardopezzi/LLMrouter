"""Abstract base provider — defines the interface all providers implement.

Providers translate normalized :class:`ChatRequest` objects into
provider-specific API calls and return normalized :class:`ChatResponse`
objects or streaming chunks.
"""

from __future__ import annotations

import abc
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from llmrouter.core.types import ChatRequest, ChatResponse, FinishReason, Usage


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, status_code: int = 500, provider: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class RetryableProviderError(ProviderError):
    """Error that can be retried (e.g. 429, 500, timeout)."""


class BaseProvider(abc.ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement :meth:`chat_completion` and
    :meth:`stream_completion`. Common functionality (retry logic,
    HTTP client management) is provided here.
    """

    def __init__(
        self,
        name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._name = name
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Provider name (e.g. ``openai``, ``ollama``)."""
        return self._name

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialized async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> BaseProvider:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        """Perform a non-streaming chat completion.

        Args:
            request: Normalized chat request.
            model: The model name to use (provider-specific).

        Returns:
            Normalized :class:`ChatResponse`.
        """
        ...

    @abc.abstractmethod
    async def stream_completion(
        self, request: ChatRequest, model: str
    ) -> AsyncIterator[dict[str, object]]:
        """Perform a streaming chat completion.

        Yields chunks in OpenAI SSE format.

        Args:
            request: Normalized chat request.
            model: The model name to use.

        Yields:
            Dictionaries representing SSE chunk payloads.
        """
        ...
        yield {}  # pragma: no cover

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_id(self, prefix: str = "chatcmpl") -> str:
        """Generate a unique response ID."""
        return f"{prefix}-{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _now() -> int:
        """Current Unix timestamp."""
        return int(time.time())

    @staticmethod
    def _make_usage(prompt_tokens: int, completion_tokens: int) -> Usage:
        """Create a :class:`Usage` instance."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    @staticmethod
    def _finish_reason(reason: str | None) -> FinishReason:
        """Map provider-specific finish reason to our enum."""
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "function_call": FinishReason.TOOL_CALLS,
        }
        if reason and reason in mapping:
            return mapping[reason]
        return FinishReason.STOP

    def _build_headers(self) -> dict[str, str]:
        """Build default headers. Override in subclasses for auth."""
        return {"Content-Type": "application/json"}
