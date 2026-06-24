"""OpenAI provider — GPT-4o, GPT-4o-mini, etc.

Uses the OpenAI-compatible base with the standard OpenAI API endpoint.
"""

from __future__ import annotations

from llmrouter.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Provider for the OpenAI API (api.openai.com)."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            name="openai",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_headers(self) -> dict[str, str]:
        """OpenAI uses Bearer token authentication."""
        headers = super()._build_headers()
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
