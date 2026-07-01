from __future__ import annotations

import httpx
import pytest

from llmrouter.config import Settings
from llmrouter.core.registry import load_model_registry
from llmrouter.core.types import ChatMessage, ChatRequest, Provider
from llmrouter.evaluator.judge import QualityJudge
from llmrouter.providers.base import ProviderError
from llmrouter.providers.ollama_provider import OllamaProvider
from llmrouter.providers.zai_provider import ZaiProvider
from llmrouter.runtime import _is_insufficient_balance_error, build_providers


@pytest.mark.asyncio
async def test_ollama_provider_uses_openai_compatible_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-ollama",
                "created": 123,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            },
        )

    provider = OllamaProvider(base_url="http://ollama.test")
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )

    response = await provider.chat_completion(
        ChatRequest(
            model=None,
            messages=[ChatMessage(role="user", content="hello")],
        ),
        "qwen2.5-coder:3b",
    )

    await provider.close()
    assert seen["path"] == "/v1/chat/completions"
    assert '"model":"qwen2.5-coder:3b"' in str(seen["body"]).replace(" ", "")
    assert response.id == "chatcmpl-ollama"
    assert response.usage.total_tokens == 4


def test_runtime_builds_ollama_without_api_key() -> None:
    registry = load_model_registry("config/models.yaml")
    providers = build_providers(Settings(), registry)

    assert Provider.OLLAMA in providers


def test_runtime_uses_ollama_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
    registry = load_model_registry("config/models.yaml")
    providers = build_providers(Settings(), registry)
    provider = providers[Provider.OLLAMA]

    assert provider._build_headers()["Authorization"] == "Bearer ollama-secret"


def test_runtime_detects_zai_insufficient_balance_error() -> None:
    error = ProviderError(
        "zai returned HTTP 429: "
        '{"error":{"code":"1113","message":"余额不足或无可用资源包,请充值。"}}',
        status_code=429,
        provider="zai",
    )

    assert _is_insufficient_balance_error(error) is True


@pytest.mark.asyncio
async def test_zai_provider_uses_current_openai_compatible_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["authorization"] = request.headers.get("authorization")
        seen["accept_language"] = request.headers.get("accept-language")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-zai",
                "created": 123,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    provider = ZaiProvider(api_key="token")
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.z.ai",
    )

    response = await provider.chat_completion(
        ChatRequest(
            model=None,
            messages=[ChatMessage(role="user", content="hello")],
        ),
        "glm-5.2",
    )

    await provider.close()
    assert seen["path"] == "/api/paas/v4/chat/completions"
    assert seen["authorization"] == "Bearer token"
    assert seen["accept_language"] == "en-US,en"
    assert response.id == "chatcmpl-zai"


@pytest.mark.asyncio
async def test_quality_judge_sends_ollama_authorization_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        content = (
            '{"relevance":5,"accuracy":5,"completeness":5,'
            '"concision":5,"safety":5,"rationale":"ok"}'
        )
        return httpx.Response(
            200,
            json={"message": {"content": content}},
        )

    judge = QualityJudge(base_url="http://ollama.test", api_key="ollama-secret")
    judge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )

    score = await judge.evaluate("prompt", "response", "model")

    await judge._client.aclose()
    assert seen["authorization"] == "Bearer ollama-secret"
    assert score.relevance == 5


@pytest.mark.asyncio
async def test_streaming_http_error_reads_error_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    provider = OllamaProvider(base_url="http://ollama.test")
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )

    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.stream_completion(
            ChatRequest(
                model=None,
                messages=[ChatMessage(role="user", content="hello")],
            ),
            "missing-model",
        ):
            pass

    await provider.close()
    assert exc_info.value.status_code == 400
    assert "bad model" in str(exc_info.value)
