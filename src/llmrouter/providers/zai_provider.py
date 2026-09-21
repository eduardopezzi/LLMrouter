"""Z.ai (Zhipu AI) provider — GLM models.

Z.ai exposes an OpenAI-compatible API endpoint.
"""

from __future__ import annotations

from llmrouter.core.types import ChatRequest
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

    def _build_payload(
        self, request: ChatRequest, model: str, *, stream: bool
    ) -> dict[str, object]:
        """Apply documented request defaults for GLM-5.3 API models.

        Both GLM-5.3 models require thinking to be enabled. GLM-5.3-Flash
        recommends top_p=0.95 and thinking.clear_thinking=false. Preserve
        caller-supplied sampling and reasoning settings when valid.
        """
        payload = super()._build_payload(request, model, stream=stream)
        if model in {"glm-5.3", "glm-5.3-flash"}:
            raw_thinking = payload.get("thinking")
            thinking = dict(raw_thinking) if isinstance(raw_thinking, dict) else {}
            thinking["type"] = "enabled"
            if model == "glm-5.3-flash":
                thinking.setdefault("clear_thinking", False)
                if not request.top_p_explicit:
                    payload["top_p"] = 0.95
            payload["thinking"] = thinking
            effort = payload.get("reasoning_effort")
            if effort not in {"low", "high", "max"}:
                payload["reasoning_effort"] = "max"
        return payload
