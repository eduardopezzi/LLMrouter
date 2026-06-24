"""NVIDIA NIM provider — Llama, Mistral, etc. via NVIDIA NIM API.

NVIDIA's NIM (NVIDIA Inference Microservices) exposes an OpenAI-compatible
endpoint at ``https://integrate.api.nvidia.com/v1``.
"""

from __future__ import annotations

from llmrouter.providers.openai_compatible import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    """Provider for the NVIDIA NIM API.

    API docs: https://docs.nvidia.com/nim/
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            name="nvidia",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_headers(self) -> dict[str, str]:
        """NVIDIA NIM uses Bearer token authentication."""
        headers = super()._build_headers()
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers["Accept"] = "application/json"
        return headers
