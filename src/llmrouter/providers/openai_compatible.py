from __future__ import annotations

import json
from collections.abc import AsyncIterator

from llmrouter.core.types import ChatRequest, ChatResponse, FinishReason, Usage
from llmrouter.providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """Base for providers that expose an OpenAI-compatible chat API."""

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        payload = self._build_payload(request, model, stream=False)
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
        payload = self._build_payload(request, model, stream=True)
        async with self.client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                elif line.startswith("data:"):
                    data = line[5:]
                else:
                    continue
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    def _build_payload(
        self, request: ChatRequest, model: str, *, stream: bool
    ) -> dict[str, object]:
        """Build the OpenAI-compatible request payload including extra fields.

        Extra fields like ``tools``, ``tool_choice``, ``response_format`` are
        passed through so that clients using function calling (e.g. Cline) work.
        """
        payload: dict[str, object] = {
            "model": model,
            "messages": [self._serialize_message(m) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        # Pass-through extra fields (tools, tool_choice, response_format, etc.)
        for key, value in request.extra.items():
            if key not in payload and value is not None:
                payload[key] = value
        return payload

    @staticmethod
    def _serialize_message(message: object) -> dict[str, object]:
        """Serialize a ChatMessage, omitting None fields for API compatibility."""
        result: dict[str, object] = {"role": message.role, "content": message.content}
        if message.name is not None:
            result["name"] = message.name
        if message.tool_calls is not None:
            result["tool_calls"] = message.tool_calls
        if message.tool_call_id is not None:
            result["tool_call_id"] = message.tool_call_id
        return result

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
