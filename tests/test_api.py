from __future__ import annotations

import json
import logging
from typing import Any

from fastapi.testclient import TestClient

from llmrouter.api.routes import create_app
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    ModelInfo,
    Provider,
    Tier,
    Usage,
)
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.feedback import FeedbackReport


class FakeProxy:
    def __init__(self) -> None:
        self.last_decision = None

    async def chat_completion(self, request: ChatRequest, decision: Any) -> ChatResponse:
        self.last_decision = decision
        return ChatResponse(
            id="chatcmpl-test",
            model=decision.primary.name,
            choices=[{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
            usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            finish_reason=FinishReason.STOP,
            created=123,
        )


class FakeStreamingProxy:
    async def stream_chat_completion(self, request: ChatRequest, decision: Any) -> Any:
        yield {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                }
            ]
        }
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class FakeFeedbackLoop:
    async def run_cycle(self, limit: int = 50) -> FeedbackReport:
        return FeedbackReport(evaluated=2, optimal=1, correct=1, overkill=0, underkill=0)


def test_chat_completions_routes_through_proxy(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="llmrouter.api")
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="cheap",
                provider=Provider.OPENAI,
                tier=Tier.T1,
                cost_per_1k_input=0.5,
                cost_per_1k_output=1.0,
            ),
        )
    )
    collector = ObservationCollector(db_path=str(tmp_path / "feedback.db"))
    app = create_app(registry=registry, proxy=FakeProxy(), collector=collector)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "say hello"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "chatcmpl-test"
    assert body["model"] == "cheap"
    assert body["llmrouter"]["selected_model"] == "cheap"
    assert len(collector._buffer) == 1
    observation = collector._buffer[0]
    assert observation.chosen_model == "cheap"
    assert observation.cost_usd == 0.002
    assert observation.metadata["provider"] == "openai"
    assert 'POST /v1/chat/completions HTTP/1.1" 200 OK' in caplog.text
    assert "selected_model=cheap" in caplog.text
    assert "provider_model=cheap" in caplog.text


def test_health_reports_model_count() -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    client = TestClient(create_app(registry=registry))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "models": 1,
        "providers": [],
        "evaluator": False,
        "openai_compatible": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "routing_roles": [],
        },
    }


def test_admin_evaluator_run_cycle() -> None:
    app = create_app(feedback_loop=FakeFeedbackLoop())
    client = TestClient(app)

    response = client.post("/admin/evaluator/run-cycle?limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "evaluated": 2,
        "optimal": 1,
        "correct": 1,
        "overkill": 0,
        "underkill": 0,
    }


def test_api_key_protects_models_endpoint() -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    client = TestClient(create_app(registry=registry, api_key="secret"))

    unauthorized = client.get("/v1/models")
    authorized = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_api_key_accepts_x_api_key_header() -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    client = TestClient(create_app(registry=registry, api_key="secret"))

    response = client.get("/v1/models", headers={"X-API-Key": "secret"})

    assert response.status_code == 200


def test_chat_completions_respects_task_role() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="summary",
                provider=Provider.OPENAI,
                tier=Tier.T1,
                capabilities=frozenset({"summarization"}),
            ),
            ModelInfo(
                name="reviewer",
                provider=Provider.OPENAI,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
            ),
        )
    )
    proxy = FakeProxy()
    app = create_app(registry=registry, proxy=proxy)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "task_role": "review",
            "messages": [{"role": "user", "content": "please review this short diff"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "reviewer"
    assert body["llmrouter"]["selected_model"] == "reviewer"


def test_streaming_chunks_are_normalized_for_openai_clients() -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    app = create_app(registry=registry, proxy=FakeStreamingProxy())
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "say hello"}],
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    first_payload = lines[0].removeprefix("data: ")
    body = json.loads(first_payload)
    assert body["object"] == "chat.completion.chunk"
    assert body["model"] == "cheap"
    assert body["choices"][0]["delta"] == {"role": "assistant", "content": "hello"}
    assert lines[-1] == "data: [DONE]"
