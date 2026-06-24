"""Ollama provider using the local OpenAI-compatible endpoint."""

from __future__ import annotations

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
