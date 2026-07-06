"""Tests for OpenAI-compatible provider HTTP error handling and payload building."""

from __future__ import annotations

from typing import Any
from collections.abc import AsyncIterator

import httpx
import pytest

from llmrouter.core.types import ChatMessage, ChatRequest, ChatResponse, FinishReason, Usage
from llmrouter.providers.base import ProviderError, RetryableProviderError
from llmrouter.providers.openai_compatible import OpenAICompatibleProvider


class TestProvider(OpenAICompatibleProvider):
    """Concrete OpenAI-compatible provider for testing."""

    def __init__(self, base_url: str = "http://test.invalid/v1", **kwargs: Any) -> None:
        super().__init__(name="test", api_key="key", base_url=base_url, **kwargs)

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        headers["Authorization"] = "Bearer test-key"
        return headers


def _request(model: str | None = None, **extra: Any) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hello")],
        extra=extra,
    )


def test_build_payload_basic() -> None:
    provider = TestProvider()
    payload = provider._build_payload(_request(), "gpt-4o", stream=False)
    assert payload["model"] == "gpt-4o"
    assert payload["stream"] is False
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0


def test_build_payload_with_max_tokens_and_stop() -> None:
    provider = TestProvider()
    request = ChatRequest(
        model=None,
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=100,
        stop=["END"],
    )
    payload = provider._build_payload(request, "model", stream=True)
    assert payload["max_tokens"] == 100
    assert payload["stop"] == ["END"]
    assert payload["stream"] is True


def test_build_payload_extra_passthrough() -> None:
    provider = TestProvider()
    request = _request(
        tools=[{"type": "function"}],
        tool_choice="auto",
        response_format={"type": "json_object"},
        seed=42,
    )
    payload = provider._build_payload(request, "model", stream=False)
    assert payload["tools"] == [{"type": "function"}]
    assert payload["tool_choice"] == "auto"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["seed"] == 42


def test_serialize_message_omits_none() -> None:
    provider = TestProvider()
    msg = ChatMessage(role="user", content="hi")
    result = provider._serialize_message(msg)
    assert result["role"] == "user"
    assert result["content"] == "hi"
    assert "name" not in result
    assert "tool_calls" not in result


def test_serialize_message_includes_optionals() -> None:
    provider = TestProvider()
    msg = ChatMessage(
        role="assistant",
        content="ok",
        name="bot",
        tool_calls=[{"id": "t1"}],
        tool_call_id="t1",
    )
    result = provider._serialize_message(msg)
    assert result["name"] == "bot"
    assert result["tool_calls"] == [{"id": "t1"}]
    assert result["tool_call_id"] == "t1"


def test_normalize_response() -> None:
    provider = TestProvider()
    body = {
        "id": "chatcmpl-123",
        "created": 1700000000,
        "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    response = provider._normalize_response(body, "gpt-4o", latency_ms=42.0)
    assert response.id == "chatcmpl-123"
    assert response.model == "gpt-4o"
    assert response.created == 1700000000
    assert response.latency_ms == 42.0
    assert response.usage.prompt_tokens == 5
    assert response.usage.total_tokens == 8
    assert response.finish_reason == FinishReason.STOP


def test_normalize_response_empty_choices() -> None:
    provider = TestProvider()
    response = provider._normalize_response({}, "model")
    assert response.id == ""
    assert response.usage.prompt_tokens == 0
    assert response.finish_reason == FinishReason.STOP


def test_normalize_response_finish_reason_length() -> None:
    provider = TestProvider()
    body = {"choices": [{"finish_reason": "length"}]}
    response = provider._normalize_response(body, "model")
    assert response.finish_reason == FinishReason.LENGTH


def test_normalize_response_finish_reason_tool_calls() -> None:
    provider = TestProvider()
    body = {"choices": [{"finish_reason": "tool_calls"}]}
    response = provider._normalize_response(body, "model")
    assert response.finish_reason == FinishReason.TOOL_CALLS


def test_normalize_response_finish_reason_function_call() -> None:
    provider = TestProvider()
    body = {"choices": [{"finish_reason": "function_call"}]}
    response = provider._normalize_response(body, "model")
    assert response.finish_reason == FinishReason.TOOL_CALLS


def test_normalize_response_created_string() -> None:
    provider = TestProvider()
    body = {"created": "1700000000"}
    response = provider._normalize_response(body, "model")
    assert response.created == 1700000000


@pytest.mark.asyncio
async def test_chat_completion_connect_error() -> None:
    provider = TestProvider(base_url="http://nonexistent.invalid:9999/v1", timeout=0.5)

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider._client = httpx.AsyncClient()
    # Patch the client's post method
    original_post = provider._client.post
    provider._client.post = mock_post  # type: ignore[assignment]

    with pytest.raises(RetryableProviderError) as exc_info:
        await provider.chat_completion(_request(), "model")
    assert exc_info.value.status_code == 503
    await provider.close()


@pytest.mark.asyncio
async def test_chat_completion_timeout_error() -> None:
    provider = TestProvider(timeout=0.5)

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    provider._client = httpx.AsyncClient()
    provider._client.post = mock_post  # type: ignore[assignment]

    with pytest.raises(RetryableProviderError) as exc_info:
        await provider.chat_completion(_request(), "model")
    assert exc_info.value.status_code == 504
    await provider.close()


@pytest.mark.asyncio
async def test_chat_completion_http_status_error() -> None:
    provider = TestProvider()

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "http://test.invalid/v1/chat/completions"),
            text="rate limited",
        )
        raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)

    provider._client = httpx.AsyncClient()
    provider._client.post = mock_post  # type: ignore[assignment]

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_request(), "model")
    assert exc_info.value.status_code == 429
    await provider.close()


@pytest.mark.asyncio
async def test_chat_completion_generic_http_error() -> None:
    provider = TestProvider()

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.HTTPError("generic error")

    provider._client = httpx.AsyncClient()
    provider._client.post = mock_post  # type: ignore[assignment]

    with pytest.raises(RetryableProviderError) as exc_info:
        await provider.chat_completion(_request(), "model")
    assert exc_info.value.status_code == 502
    await provider.close()


@pytest.mark.asyncio
async def test_chat_completion_success() -> None:
    provider = TestProvider()

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "id": "test-id",
                "created": 1700000000,
                "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
            request=httpx.Request("POST", "http://test.invalid/v1/chat/completions"),
        )

    provider._client = httpx.AsyncClient()
    provider._client.post = mock_post  # type: ignore[assignment]

    response = await provider.chat_completion(_request(), "gpt-4o")
    assert response.id == "test-id"
    assert response.usage.prompt_tokens == 5
    await provider.close()


@pytest.mark.asyncio
async def test_provider_close_and_aenter_aexit() -> None:
    provider = TestProvider()
    _ = provider.client  # Force lazy init
    async with provider as p:
        assert p is provider
    assert provider._client is None