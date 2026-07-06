"""Tests for API route helper functions and streaming logic."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from llmrouter.api.routes import (
    ChatCompletionPayload,
    create_app,
    _to_chat_request,
    _chat_request_directives,
    _prompt_directives,
    _with_prompt_directives,
    _infer_project_from_prompt,
    _retrieve_memory,
    _with_memory_context,
    _record_memory,
    _memory_disabled,
    _memory_project,
    _memory_default_project,
    _normalize_stream_chunk,
    _chunk_has_assistant_output,
    _extract_delta_text,
    _model_payload,
    _routing_constraints,
    _routing_roles,
    _estimate_cost,
    _task_role,
    _precog_project,
    _rag_metadata,
    _memory_payload,
    _prompt_hash,
    _choice_text,
    LLMrouterFeedbackPayload,
)
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingStrategy,
    Tier,
    Usage,
)
from llmrouter.memory import MemoryConfig, MemoryEntry, SQLiteMemoryStore


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(name="m1", provider=Provider.OPENAI, tier=Tier.T1, capabilities=frozenset({"code"})),
            ModelInfo(name="m2", provider=Provider.OLLAMA, tier=Tier.T2, capabilities=frozenset({"review"})),
        )
    )


# ---------------------------------------------------------------------------
# Directives parsing
# ---------------------------------------------------------------------------


def test_prompt_directives_project() -> None:
    result = _prompt_directives("{{project: my-project}}\nhello")
    assert result["project"] == "my-project"


def test_prompt_directives_task_role() -> None:
    result = _prompt_directives("{{role: review}}\nhello")
    assert result["task_role"] == "review"


def test_prompt_directives_model() -> None:
    result = _prompt_directives("{{model: cheap}}\nhello")
    assert result["model"] == "cheap"


def test_prompt_directives_alias_p() -> None:
    result = _prompt_directives("{{p: proj}}\nhello")
    assert result["project"] == "proj"


def test_prompt_directives_alias_t() -> None:
    result = _prompt_directives("{{t: fix}}\nhello")
    assert result["task_role"] == "fix"


def test_prompt_directives_alias_m() -> None:
    result = _prompt_directives("{{m: model-x}}\nhello")
    assert result["model"] == "model-x"


def test_prompt_directives_empty() -> None:
    assert _prompt_directives("just a normal prompt") == {}


def test_prompt_directives_list_content() -> None:
    result = _prompt_directives([{"text": "{{project: list-proj}}"}])
    assert result["project"] == "list-proj"


def test_prompt_directives_quoted_value() -> None:
    result = _prompt_directives('{{project: "quoted"}}')
    assert result["project"] == "quoted"


def test_chat_request_directives() -> None:
    req = ChatRequest(
        model=None,
        messages=[ChatMessage(role="user", content="{{project: test}}\nhello")],
    )
    directives = _chat_request_directives(req)
    assert directives["project"] == "test"


def test_with_prompt_directives_model() -> None:
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    from llmrouter.api.routes import ChatCompletionPayload
    # Can't easily construct payload, test with empty directives
    result = _with_prompt_directives(req, MagicMock(model=None, extra={}), {})
    assert result is req  # No directives → no change


def test_with_prompt_directives_model_from_directive() -> None:
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    from llmrouter.api.routes import ChatCompletionPayload
    payload_mock = MagicMock(model="auto", extra={})
    result = _with_prompt_directives(req, payload_mock, {"model": "cheap"})
    assert result.model == "cheap"


def test_with_prompt_directives_auto_from_directive() -> None:
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    payload_mock = MagicMock(model="auto", extra={})
    result = _with_prompt_directives(req, payload_mock, {"model": "auto"})
    assert result.model is None


# ---------------------------------------------------------------------------
# Project inference
# ---------------------------------------------------------------------------


def test_infer_project_from_prompt_workspace() -> None:
    result = _infer_project_from_prompt("Current Workspace Directory (/home/user/MyProject)")
    assert result == "MyProject"


def test_infer_project_from_prompt_github() -> None:
    result = _infer_project_from_prompt("Working in /github/MyRepo issues")
    assert result == "MyRepo"


def test_infer_project_from_prompt_workspace_colon() -> None:
    result = _infer_project_from_prompt("workspace: `my-project`")
    assert result == "my-project"


def test_infer_project_from_prompt_empty() -> None:
    assert _infer_project_from_prompt("") is None


def test_infer_project_from_prompt_no_match() -> None:
    assert _infer_project_from_prompt("just a regular question") is None


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def test_memory_default_project_with_store() -> None:
    config = MemoryConfig(enabled=True, default_project="custom")
    store = SQLiteMemoryStore(config)
    assert _memory_default_project(store, "fallback") == "custom"


def test_memory_default_project_without_store() -> None:
    assert _memory_default_project(None, "fallback") == "fallback"


def test_memory_project_from_header() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload(messages=[ChatMessage(role="user", content="hi")] if False else [])
    # Construct minimal payload
    payload = ChatCompletionPayload.model_validate({"messages": [{"role": "user", "content": "hi"}]})

    request_mock = MagicMock()
    request_mock.headers = {"x-llmrouter-project": "header-proj"}
    result = _memory_project(payload, request_mock, default="def")
    assert result == "header-proj"


def test_memory_project_from_llmrouter() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"project": "router-proj"},
    })
    request_mock = MagicMock()
    request_mock.headers = {}
    result = _memory_project(payload, request_mock, default="def")
    assert result == "router-proj"


def test_memory_project_from_metadata() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"project": "meta-proj"},
    })
    request_mock = MagicMock()
    request_mock.headers = {}
    result = _memory_project(payload, request_mock, default="def")
    assert result == "meta-proj"


def test_memory_project_from_directive() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    request_mock = MagicMock()
    request_mock.headers = {}
    result = _memory_project(payload, request_mock, default="def", directives={"project": "dir-proj"})
    assert result == "dir-proj"


def test_memory_project_from_inferred() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "Workspace (/github/InferredProj)"}],
    })
    request_mock = MagicMock()
    request_mock.headers = {}
    result = _memory_project(payload, request_mock, default="def", prompt="Workspace (/github/InferredProj)")
    assert result == "InferredProj"


def test_memory_project_default() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    request_mock = MagicMock()
    request_mock.headers = {}
    result = _memory_project(payload, request_mock, default="def")
    assert result == "def"


def test_memory_disabled_dict() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"memory": {"enabled": False}},
    })
    assert _memory_disabled(payload) is True


def test_memory_disabled_bool() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"memory": False},
    })
    assert _memory_disabled(payload) is True


def test_memory_not_disabled() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert _memory_disabled(payload) is False


def test_retrieve_memory_disabled() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"memory": {"enabled": False}},
    })
    req = _to_chat_request(payload)
    assert _retrieve_memory(None, project="p", chat_request=req, payload=payload) == []


def test_retrieve_memory_no_store() -> None:
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    req = _to_chat_request(payload)
    assert _retrieve_memory(None, project="p", chat_request=req, payload=payload) == []


def test_with_memory_context_no_store() -> None:
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content="hi")])
    assert _with_memory_context(req, [], memory_store=None) is req


def test_with_memory_context_empty_entries() -> None:
    config = MemoryConfig(enabled=True)
    store = SQLiteMemoryStore(config)
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content="hi")])
    assert _with_memory_context(req, [], memory_store=store) is req


def test_with_memory_context_with_entries() -> None:
    config = MemoryConfig(enabled=True, max_context_chars=500)
    store = SQLiteMemoryStore(config)
    entry = MemoryEntry(id=1, project="p", prompt="test prompt", response="test response", score=0.9)
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content="hi")])
    result = _with_memory_context(req, [entry], memory_store=store)
    assert len(result.messages) == 2  # system + original
    assert result.messages[0].role == "system"
    assert result.extra.get("llmrouter_memory", {}).get("used") is True


# ---------------------------------------------------------------------------
# Stream chunk helpers
# ---------------------------------------------------------------------------


def test_normalize_stream_chunk_valid() -> None:
    chunk = {"id": "123", "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
    result = _normalize_stream_chunk(chunk, "model-x")
    assert result is not None
    assert result["model"] == "model-x"
    assert result["choices"][0]["delta"]["content"] == "hi"


def test_normalize_stream_chunk_no_choices() -> None:
    assert _normalize_stream_chunk({"choices": []}, "model") is None


def test_normalize_stream_chunk_not_list() -> None:
    assert _normalize_stream_chunk({"choices": "bad"}, "model") is None


def test_normalize_stream_chunk_with_message() -> None:
    chunk = {"choices": [{"message": {"content": "hello", "role": "assistant"}}]}
    result = _normalize_stream_chunk(chunk, "model")
    assert result is not None
    assert result["choices"][0]["delta"]["content"] == "hello"


def test_normalize_stream_chunk_defaults() -> None:
    chunk = {"choices": [{}]}
    result = _normalize_stream_chunk(chunk, "model")
    assert result is not None
    assert result["object"] == "chat.completion.chunk"


def test_chunk_has_assistant_output_content() -> None:
    chunk = {"choices": [{"delta": {"content": "hi"}}]}
    assert _chunk_has_assistant_output(chunk) is True


def test_chunk_has_assistant_output_tool_calls() -> None:
    chunk = {"choices": [{"delta": {"tool_calls": [{"id": "t1"}]}}]}
    assert _chunk_has_assistant_output(chunk) is True


def test_chunk_has_assistant_output_empty() -> None:
    chunk = {"choices": [{"delta": {}}]}
    assert _chunk_has_assistant_output(chunk) is False


def test_chunk_has_assistant_output_no_choices() -> None:
    assert _chunk_has_assistant_output({"choices": []}) is False


def test_extract_delta_text_from_delta() -> None:
    chunk = {"choices": [{"delta": {"content": "hello"}}]}
    acc: list[str] = []
    _extract_delta_text(chunk, acc)
    assert acc == ["hello"]


def test_extract_delta_text_from_message() -> None:
    chunk = {"choices": [{"message": {"content": "world"}}]}
    acc: list[str] = []
    _extract_delta_text(chunk, acc)
    assert acc == ["world"]


def test_extract_delta_text_empty() -> None:
    chunk = {"choices": []}
    acc: list[str] = []
    _extract_delta_text(chunk, acc)
    assert acc == []


# ---------------------------------------------------------------------------
# Model payload and routing
# ---------------------------------------------------------------------------


def test_model_payload() -> None:
    model = ModelInfo(name="test", provider=Provider.OPENAI, tier=Tier.T2, capabilities=frozenset({"code"}))
    result = _model_payload(model)
    assert result["id"] == "test"
    assert result["owned_by"] == "openai"
    assert result["llmrouter"]["tier"] == 2


def test_routing_constraints_with_role() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "task_role": "review",
    })
    constraints = _routing_constraints(payload)
    assert "review" in constraints.required_capabilities


def test_routing_constraints_no_role() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    constraints = _routing_constraints(payload)
    assert constraints.required_capabilities == frozenset()


def test_routing_constraints_from_router_options() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"task_role": "fix"},
    })
    constraints = _routing_constraints(payload)
    assert "fix" in constraints.required_capabilities


def test_routing_constraints_from_directives() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    constraints = _routing_constraints(payload, directives={"task_role": "code"})
    assert "code" in constraints.required_capabilities


def test_routing_roles() -> None:
    roles = _routing_roles(_registry())
    assert "code" in roles
    assert "review" in roles


# ---------------------------------------------------------------------------
# Cost and misc helpers
# ---------------------------------------------------------------------------


def test_estimate_cost() -> None:
    model = ModelInfo(name="m", provider=Provider.OPENAI, tier=Tier.T1, cost_per_1k_input=0.01, cost_per_1k_output=0.02)
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = _estimate_cost(model, usage)
    assert cost == pytest.approx(0.01 + 0.01)


def test_task_role_from_payload() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "task_role": "review",
    })
    assert _task_role(payload) == "review"


def test_task_role_from_router() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"role": "fix"},
    })
    assert _task_role(payload) == "fix"


def test_task_role_empty() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert _task_role(payload) == ""


def test_precog_project() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"project": "precog-proj"},
    })
    assert _precog_project(payload, "default") == "precog-proj"


def test_precog_project_default() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert _precog_project(payload, "default") == "default"


def test_rag_metadata_not_used() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
    })
    result = _rag_metadata(payload)
    assert result["used"] is False


def test_rag_metadata_used() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "llmrouter": {"rag": {"used": True, "collection": "docs", "top_k": 5, "context_tokens": 100}},
    })
    result = _rag_metadata(payload)
    assert result["used"] is True
    assert result["collection"] == "docs"
    assert result["top_k"] == 5


def test_memory_payload() -> None:
    entries = [MemoryEntry(id=1, project="p", prompt="x", response="y", score=0.5)]
    result = _memory_payload(entries, "proj")
    assert result["used"] is True
    assert result["top_k"] == 1
    assert result["ids"] == [1]


def test_memory_payload_empty() -> None:
    result = _memory_payload([], "proj")
    assert result["used"] is False


def test_prompt_hash() -> None:
    h1 = _prompt_hash("hello")
    h2 = _prompt_hash("hello")
    h3 = _prompt_hash("world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA256 hex


def test_choice_text_from_message() -> None:
    assert _choice_text({"message": {"content": "hello"}}) == "hello"


def test_choice_text_from_text() -> None:
    assert _choice_text({"text": "fallback"}) == "fallback"


def test_choice_text_empty() -> None:
    assert _choice_text({}) == ""


# ---------------------------------------------------------------------------
# to_chat_request
# ---------------------------------------------------------------------------


def test_to_chat_request_basic() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
    })
    req = _to_chat_request(payload)
    assert req.model == "test-model"
    assert len(req.messages) == 1
    assert req.messages[0].content == "hello"


def test_to_chat_request_with_content_list() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}],
    })
    req = _to_chat_request(payload)
    assert "part1" in req.messages[0].content
    assert "part2" in req.messages[0].content


def test_to_chat_request_with_extra_fields() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "test"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "seed": 42,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "n": 1,
        "logit_bias": {"token": 1.0},
        "user": "user123",
        "task_role": "review",
        "metadata": {"key": "value"},
        "llmrouter": {"project": "proj"},
    })
    req = _to_chat_request(payload)
    assert req.extra["tools"] == [{"type": "function", "function": {"name": "test"}}]
    assert req.extra["tool_choice"] == "auto"
    assert req.extra["seed"] == 42
    assert req.extra["task_role"] == "review"
    assert req.extra["metadata"] == {"key": "value"}
    assert req.extra["llmrouter"] == {"project": "proj"}


def test_to_chat_request_max_completion_tokens() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 200,
    })
    req = _to_chat_request(payload)
    assert req.max_tokens == 200


def test_to_chat_request_stop_string() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END",
    })
    req = _to_chat_request(payload)
    assert req.stop == ["END"]


def test_to_chat_request_stop_list() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["END", "STOP"],
    })
    req = _to_chat_request(payload)
    assert req.stop == ["END", "STOP"]


def test_to_chat_request_none_content() -> None:
    from llmrouter.api.routes import ChatCompletionPayload
    payload = ChatCompletionPayload.model_validate({
        "messages": [{"role": "user", "content": None}],
    })
    req = _to_chat_request(payload)
    assert req.messages[0].content == ""