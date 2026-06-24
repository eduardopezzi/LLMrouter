from __future__ import annotations

from collections.abc import AsyncIterator

from llmrouter.core.types import ChatRequest, ChatResponse, FinishReason, Usage
from llmrouter.providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """Base for providers that expose an OpenAI-compatible chat API."""

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        payload: dict[str, object] = {
            "model": model,
            "messages": [message.__dict__ for message in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop

        response = await self.client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        body = response.json()
        return self._normalize_response(body, model)

    async def stream_completion(
        self, request: ChatRequest, model: str
    ) -> AsyncIterator[dict[str, object]]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [message.__dict__ for message in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop

        async with self.client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                if chunk:
                    yield {"data": chunk}

    def _normalize_response(self, body: dict[str, object], model: str) -> ChatResponse:
        choices = body.get("choices", [])
        metadata = body.get("usage", {})
        usage = Usage(
            prompt_tokens=int(metadata.get("prompt_tokens", 0)),
            completion_tokens=int(metadata.get("completion_tokens", 0)),
            total_tokens=int(metadata.get("total_tokens", 0)),
        )
        finish_reason = FinishReason.STOP
        if choices and isinstance(choices[0], dict):
            finish_reason = self._finish_reason(choices[0].get("finish_reason"))
        return ChatResponse(
            id=body.get("id", ""),
            model=model,
            choices=choices,
            usage=usage,
            finish_reason=finish_reason,
            created=int(body.get("created", 0) or 0),
        )
