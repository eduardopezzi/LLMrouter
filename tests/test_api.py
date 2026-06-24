from __future__ import annotations

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
    async def chat_completion(self, request: ChatRequest, decision: object) -> ChatResponse:
        return ChatResponse(
            id="chatcmpl-test",
            model="cheap",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
            usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            finish_reason=FinishReason.STOP,
            created=123,
        )


class FakeFeedbackLoop:
    async def run_cycle(self, limit: int = 50) -> FeedbackReport:
        return FeedbackReport(evaluated=2, optimal=1, correct=1, overkill=0, underkill=0)


def test_chat_completions_routes_through_proxy(tmp_path) -> None:
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
