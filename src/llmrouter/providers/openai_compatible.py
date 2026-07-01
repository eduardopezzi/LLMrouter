from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from llmrouter.core.types import ChatRequest, ChatResponse, FinishReason, Usage
from llmrouter.logging_config import get_logger
from llmrouter.providers.base import BaseProvider, ProviderError, RetryableProviderError

_logger = get_logger("llmrouter.provider")


class OpenAICompatibleProvider(BaseProvider):
    """Base for providers that expose an OpenAI-compatible chat API."""

    async def chat_completion(self, request: ChatRequest, model: str) -> ChatResponse:
        payload = self._build_payload(request, model, stream=False)
        url = f"{self._base_url}/chat/completions"
        _logger.debug(
            "%s → POST %s | model=%s, messages=%d",
            self._name,
            url,
            model,
            len(request.messages),
        )
        started = time.perf_counter()
        try:
            response = await self.client.post(
                url,
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            _logger.debug(
                "%s connection FAILED in %.0fms",
                self._name,
                (time.perf_counter() - started) * 1000,
            )
            raise RetryableProviderError(
                f"Could not connect to {self._name} at {self._base_url}: {exc}",
                status_code=503,
                provider=self._name,
            ) from exc
        except httpx.TimeoutException as exc:
            _logger.debug(
                "%s TIMEOUT after %.0fms",
                self._name,
                (time.perf_counter() - started) * 1000,
            )
            raise RetryableProviderError(
                f"Request to {self._name} timed out after {self._timeout}s: {exc}",
                status_code=504,
                provider=self._name,
            ) from exc
        except httpx.HTTPStatusError as exc:
            _logger.debug(
                "%s HTTP %d in %.0fms",
                self._name,
                exc.response.status_code,
                (time.perf_counter() - started) * 1000,
            )
            raise ProviderError(
                f"{self._name} returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                status_code=exc.response.status_code,
                provider=self._name,
            ) from exc
        except httpx.HTTPError as exc:
            raise RetryableProviderError(
                f"Transport error contacting {self._name}: {exc}",
                status_code=502,
                provider=self._name,
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        _logger.debug("%s ← HTTP 200 in %.0fms", self._name, elapsed_ms)
        body = response.json()
        return self._normalize_response(body, model)

    async def stream_completion(
        self, request: ChatRequest, model: str
    ) -> AsyncIterator[dict[str, object]]:
        payload = self._build_payload(request, model, stream=True)
        url = f"{self._base_url}/chat/completions"
        _logger.debug(
            "%s → POST (stream) %s | model=%s, messages=%d",
            self._name,
            url,
            model,
            len(request.messages),
        )
        try:
            async with self.client.stream(
                "POST",
                url,
                json=payload,
                headers=self._build_headers(),
            ) as response:
                if response.status_code >= 400:
                    error_body = (await response.aread()).decode(errors="replace")
                    raise ProviderError(
                        f"{self._name} returned HTTP {response.status_code}: {error_body[:500]}",
                        status_code=response.status_code,
                        provider=self._name,
                    )
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
                    except GeneratorExit:
                        # Client disconnected — close stream cleanly
                        _logger.debug("%s stream: client disconnected", self._name)
                        return
        except GeneratorExit:
            # Generator was garbage-collected or closed — exit silently
            _logger.debug("%s stream: generator closed", self._name)
            return
        except httpx.ConnectError as exc:
            raise RetryableProviderError(
                f"Could not connect to {self._name} at {self._base_url}: {exc}",
                status_code=503,
                provider=self._name,
            ) from exc
        except httpx.TimeoutException as exc:
            raise RetryableProviderError(
                f"Stream request to {self._name} timed out after {self._timeout}s: {exc}",
                status_code=504,
                provider=self._name,
            ) from exc
        except httpx.HTTPError as exc:
            raise RetryableProviderError(
                f"Stream interrupted from {self._name}: {exc}",
                status_code=502,
                provider=self._name,
            ) from exc

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
