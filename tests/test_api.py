from __future__ import annotations

import json
import logging
from typing import Any

import pytest
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
from llmrouter.memory import MemoryConfig, SQLiteMemoryStore


class FakeProxy:
    def __init__(self) -> None:
        self.last_request = None
        self.last_decision = None

    async def chat_completion(self, request: ChatRequest, decision: Any) -> ChatResponse:
        self.last_request = request
        self.last_decision = decision
        return ChatResponse(
            id="chatcmpl-test",
            model=decision.primary.name,
            choices=[{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
            usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            finish_reason=FinishReason.STOP,
            created=123,
            latency_ms=50.0,
        )


class FakeStreamingProxy:
    def __init__(self) -> None:
        self.last_request = None
        self.last_decision = None

    async def stream_chat_completion(self, request: ChatRequest, decision: Any) -> Any:
        self.last_request = request
        self.last_decision = decision
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


class FakePrecogPublisher:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.updates: list[tuple[str, dict[str, Any]]] = []

    def record_observation(self, payload: dict[str, Any]) -> None:
        self.observations.append(payload)

    def update_observation(self, request_id: str, outcome: dict[str, Any]) -> None:
        self.updates.append((request_id, outcome))


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


@pytest.mark.parametrize(
    ("payload_patch", "expected"),
    [
        (
            {
                "messages": [
                    {"role": "user", "content": "use a tool"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
                ],
            },
            {"message_index": 1, "content": "", "has_tool_calls": True},
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "first block"},
                            "second block",
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            },
                        ],
                    }
                ],
            },
            {"message_index": 0, "content": "first block\nsecond block"},
        ),
        ({"stop": "</tool_call>"}, {"stop": ["</tool_call>"]}),
        ({"temperature": None, "top_p": None}, {"temperature": 1.0, "top_p": 1.0}),
        (
            {
                "max_completion_tokens": 128,
                "parallel_tool_calls": True,
                "reasoning_effort": "low",
            },
            {
                "max_tokens": 128,
                "extra": {
                    "parallel_tool_calls": True,
                    "reasoning_effort": "low",
                },
            },
        ),
    ],
)
def test_chat_completions_accepts_cline_openai_compatible_variants(
    payload_patch: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    proxy = FakeProxy()
    app = create_app(registry=registry, proxy=proxy)
    client = TestClient(app)
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "use a tool"}],
        "stream": False,
    }
    payload.update(payload_patch)

    response = client.post(
        "/v1/chat/completions",
        json=payload,
    )

    assert response.status_code == 200
    assert proxy.last_request is not None
    if "message_index" in expected:
        message = proxy.last_request.messages[expected["message_index"]]
        assert message.content == expected["content"]
        if expected.get("has_tool_calls"):
            assert message.tool_calls is not None
    if "stop" in expected:
        assert proxy.last_request.stop == expected["stop"]
    if "temperature" in expected:
        assert proxy.last_request.temperature == expected["temperature"]
    if "top_p" in expected:
        assert proxy.last_request.top_p == expected["top_p"]
    if "max_tokens" in expected:
        assert proxy.last_request.max_tokens == expected["max_tokens"]
    for key, value in expected.get("extra", {}).items():
        assert proxy.last_request.extra[key] == value


def test_chat_completions_publishes_precog_observation() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="cheap",
                provider=Provider.DEEPSEEK,
                tier=Tier.T1,
                cost_per_1k_input=0.5,
                cost_per_1k_output=1.0,
            ),
        )
    )
    publisher = FakePrecogPublisher()
    app = create_app(
        registry=registry,
        proxy=FakeProxy(),
        precog_publisher=publisher,
        precog_project="llmrouter-tests",
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "req-123"},
        json={
            "task_role": "fix",
            "messages": [{"role": "user", "content": "corrija este arquivo"}],
            "llmrouter": {
                "project": "precog",
                "rag": {
                    "used": True,
                    "collection": "project_docs",
                    "top_k": 3,
                    "context_tokens": 120,
                },
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["llmrouter"]["request_id"] == "req-123"
    assert body["llmrouter"]["provider"] == "deepseek"
    assert publisher.observations == [
        {
            "request_id": "req-123",
            "project": "precog",
            "task_role": "fix",
            "prompt_hash": publisher.observations[0]["prompt_hash"],
            "selected_model": "cheap",
            "provider": "deepseek",
            "provider_model": "cheap",
            "latency_ms": publisher.observations[0]["latency_ms"],
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "cost_usd": 0.002,
            "rag": {
                "used": True,
                "collection": "project_docs",
                "top_k": 3,
                "context_tokens": 120,
            },
            "memory": {
                "used": False,
                "project": "precog",
                "top_k": 0,
                "ids": [],
            },
        }
    ]
    assert len(publisher.observations[0]["prompt_hash"]) == 64


def test_llmrouter_feedback_forwards_to_precog() -> None:
    publisher = FakePrecogPublisher()
    app = create_app(precog_publisher=publisher)
    client = TestClient(app)

    response = client.post(
        "/v1/llmrouter/feedback",
        json={
            "request_id": "req-123",
            "outcome": {
                "accepted": True,
                "tests_passed": True,
                "validated": True,
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "request_id": "req-123"}
    assert publisher.updates == [
        (
            "req-123",
            {
                "accepted": True,
                "tests_passed": True,
                "validated": True,
            },
        )
    ]


def test_llmrouter_feedback_requires_precog_publisher() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/llmrouter/feedback",
        json={"request_id": "req-123", "outcome": {"accepted": True}},
    )

    assert response.status_code == 503


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
        "memory": False,
        "health_tracker": False,
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


def test_chat_completions_accepts_prompt_directives() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(name="summary", provider=Provider.OPENAI, tier=Tier.T1),
            ModelInfo(
                name="reviewer",
                provider=Provider.OPENAI,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
            ),
            ModelInfo(name="specialist", provider=Provider.OPENAI, tier=Tier.T3),
        )
    )
    proxy = FakeProxy()
    app = create_app(registry=registry, proxy=proxy)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "{{project:PRecog}} {{task:review}} {{model:specialist}}\n"
                        "Review this implementation."
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["llmrouter"]["selected_model"] == "specialist"
    assert body["llmrouter"]["memory"]["project"] == "PRecog"
    assert proxy.last_request is not None
    assert proxy.last_request.model == "specialist"
    assert proxy.last_request.extra["llmrouter_prompt_directives"] == {
        "project": "PRecog",
        "task_role": "review",
        "model": "specialist",
    }


def test_chat_completions_accepts_prompt_directives_after_context_messages() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(name="summary", provider=Provider.OPENAI, tier=Tier.T1),
            ModelInfo(name="specialist", provider=Provider.OPENAI, tier=Tier.T3),
        )
    )
    proxy = FakeProxy()
    app = create_app(registry=registry, proxy=proxy)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Large client context.\n" * 10},
                {"role": "assistant", "content": "Previous assistant response.\n" * 10},
                {
                    "role": "user",
                    "content": (
                        "{{project:PRecog}} {{task:refactoring}} {{model:specialist}}\n"
                        "Refactor this module."
                    ),
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["llmrouter"]["selected_model"] == "specialist"
    assert body["llmrouter"]["memory"]["project"] == "PRecog"
    assert proxy.last_request is not None
    assert proxy.last_request.model == "specialist"


def test_chat_completions_records_and_injects_project_memory(tmp_path) -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    proxy = FakeProxy()
    memory_store = SQLiteMemoryStore(
        MemoryConfig(
            enabled=True,
            db_path=str(tmp_path / "memory.db"),
            min_prompt_chars=1,
            min_response_chars=1,
            min_score=0.05,
        )
    )
    app = create_app(registry=registry, proxy=proxy, memory_store=memory_store)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "llmrouter": {"project": "alpha"},
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Remember the alpha project database schema uses users.email UUID keys."
                    ),
                }
            ],
        },
    )
    assert first.status_code == 200
    assert first.json()["llmrouter"]["memory"]["used"] is False

    second = client.post(
        "/v1/chat/completions",
        json={
            "llmrouter": {"project": "alpha"},
            "messages": [
                {
                    "role": "user",
                    "content": "For alpha, what did we decide about users email schema keys?",
                }
            ],
        },
    )

    assert second.status_code == 200
    assert second.json()["llmrouter"]["memory"]["used"] is True
    assert proxy.last_request is not None
    assert proxy.last_request.messages[0].role == "system"
    assert "Relevant project memory" in str(proxy.last_request.messages[0].content)
    assert "users.email UUID keys" in str(proxy.last_request.messages[0].content)

    third = client.post(
        "/v1/chat/completions",
        json={
            "llmrouter": {"project": "beta"},
            "messages": [
                {
                    "role": "user",
                    "content": "For beta, what did we decide about users email schema keys?",
                }
            ],
        },
    )

    assert third.status_code == 200
    assert third.json()["llmrouter"]["memory"]["used"] is False


def test_chat_completions_infers_memory_project_from_workspace_prompt(tmp_path) -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    proxy = FakeProxy()
    memory_store = SQLiteMemoryStore(
        MemoryConfig(
            enabled=True,
            db_path=str(tmp_path / "memory.db"),
            min_prompt_chars=1,
            min_response_chars=1,
            min_score=0.05,
        )
    )
    app = create_app(registry=registry, proxy=proxy, memory_store=memory_store)
    client = TestClient(app)

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Current Workspace Directory (/Users/me/github/PRecog)\n"
                        "Remember Phoenix service owns contracts."
                    ),
                }
            ],
        },
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Current Workspace Directory (/Users/me/github/PRecog)\n"
                        "Who owns contracts?"
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["llmrouter"]["memory"]["project"] == "PRecog"
    assert response.json()["llmrouter"]["memory"]["used"] is True


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


def test_streaming_chat_completions_accepts_cline_tool_call_payload() -> None:
    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    proxy = FakeStreamingProxy()
    app = create_app(registry=registry, proxy=proxy)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "use a tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
            ],
            "stream": True,
            "temperature": None,
            "top_p": None,
            "stop": "</tool_call>",
            "max_completion_tokens": 128,
            "parallel_tool_calls": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert proxy.last_request is not None
    assert proxy.last_request.messages[1].content == ""
    assert proxy.last_request.messages[1].tool_calls is not None
    assert proxy.last_request.stop == ["</tool_call>"]
    assert proxy.last_request.max_tokens == 128
    assert proxy.last_request.temperature == 1.0
    assert proxy.last_request.top_p == 1.0
    assert proxy.last_request.extra["parallel_tool_calls"] is True
    assert lines[-1] == "data: [DONE]"
