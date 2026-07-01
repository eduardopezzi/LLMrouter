"""Z.ai (Zhipu AI) provider — GLM-4, ChatGLM, etc.

Z.ai exposes an OpenAI-compatible API endpoint.
"""

from __future__ import annotations

from llmrouter.providers.openai_compatible import OpenAICompatibleProvider


class ZaiProvider(OpenAICompatibleProvider):
    """Provider for the Z.ai (Zhipu AI) API.

    API docs: https://docs.z.ai/
    """

    DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            name="zai",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_headers(self) -> dict[str, str]:
        """Z.ai uses Bearer token authentication."""
        headers = super()._build_headers()
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers["Accept-Language"] = "en-US,en"
        return headers
