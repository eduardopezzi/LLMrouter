"""Ollama provider using the local OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any

import httpx

from llmrouter.providers.base import ProviderError, RetryableProviderError
from llmrouter.providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Provider for a local Ollama server."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        resolved_base_url = base_url or self.DEFAULT_BASE_URL
        resolved_base_url = resolved_base_url.rstrip("/")
        if not resolved_base_url.endswith("/v1"):
            resolved_base_url = f"{resolved_base_url}/v1"
        super().__init__(
            name="ollama",
            api_key=api_key,
            base_url=resolved_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def embeddings(
        self,
        *,
        model: str,
        inputs: list[str],
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings with Ollama's native batch API.

        Ollama's chat API is OpenAI-compatible, but embeddings use its native
        ``/api/embed`` route. LLMrouter normalizes that response at its public
        OpenAI-compatible ``/v1/embeddings`` endpoint.
        """
        native_base_url = self._base_url.removesuffix("/v1")
        payload: dict[str, Any] = {"model": model, "input": inputs, "truncate": truncate}
        if dimensions is not None:
            payload["dimensions"] = dimensions

        try:
            response = await self.client.post(
                f"{native_base_url}/api/embed",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RetryableProviderError(
                f"Could not connect to Ollama at {native_base_url}: {exc}",
                status_code=503,
                provider=self._name,
            ) from exc
        except httpx.TimeoutException as exc:
            raise RetryableProviderError(
                f"Ollama embedding request timed out after {self._timeout}s: {exc}",
                status_code=504,
                provider=self._name,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                status_code=exc.response.status_code,
                provider=self._name,
            ) from exc
        except httpx.HTTPError as exc:
            raise RetryableProviderError(
                f"Transport error contacting Ollama embeddings: {exc}",
                status_code=502,
                provider=self._name,
            ) from exc

        body = response.json()
        raw_embeddings = body.get("embeddings") if isinstance(body, dict) else None
        if (
            not isinstance(raw_embeddings, list)
            or len(raw_embeddings) != len(inputs)
            or any(not isinstance(vector, list) or not vector for vector in raw_embeddings)
        ):
            raise ProviderError(
                "Ollama returned an invalid embedding batch",
                status_code=502,
                provider=self._name,
            )
        try:
            embeddings = [[float(value) for value in vector] for vector in raw_embeddings]
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "Ollama returned a non-numeric embedding value",
                status_code=502,
                provider=self._name,
            ) from exc

        prompt_tokens = body.get("prompt_eval_count", 0)
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            prompt_tokens = 0
        return embeddings, max(0, prompt_tokens)
