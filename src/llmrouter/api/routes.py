"""FastAPI routes for the OpenAI-compatible gateway."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    RoutingConstraints,
    RoutingStrategy,
    Usage,
)
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.feedback import FeedbackLoop
from llmrouter.evaluator.types import RoutingObservation
from llmrouter.logging_config import get_logger
from llmrouter.providers.base import ProviderError

_logger = get_logger("llmrouter.api")


class ChatMessagePayload(BaseModel):
    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionPayload(BaseModel):
    model: str | None = None
    messages: list[ChatMessagePayload]
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    top_p: float = 1.0
    stop: list[str] | None = None
    # Pass-through fields for OpenAI-compatible clients (Cline, etc.)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    task_role: str | None = Field(
        default=None,
        description="Optional LLMrouter routing role, e.g. review, test_generation, fix.",
    )
    metadata: dict[str, Any] | None = None
    llmrouter: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMrouterFeedbackPayload(BaseModel):
    """Post-execution feedback for an LLMrouter request."""

    request_id: str = Field(min_length=1, max_length=128)
    outcome: dict[str, Any] = Field(default_factory=dict)


def create_app(
    *,
    registry: ModelRegistry | None = None,
    router: MultiModelRouter | None = None,
    proxy: ProviderProxy | None = None,
    collector: ObservationCollector | None = None,
    feedback_loop: FeedbackLoop | None = None,
    evaluator_interval_seconds: int | None = None,
    api_key: str | None = None,
    cors_origins: list[str] | None = None,
    precog_publisher: Any | None = None,
    precog_project: str = "llmrouter",
) -> FastAPI:
    """Build the FastAPI application with injectable runtime components."""
    model_registry = registry or ModelRegistry()
    app_router = router or MultiModelRouter(
        model_registry,
        PromptScorer(),
        RoutingStrategy.COST,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        worker: asyncio.Task[None] | None = None
        if feedback_loop is not None and evaluator_interval_seconds:
            worker = asyncio.create_task(
                _run_feedback_worker(feedback_loop, evaluator_interval_seconds)
            )
        try:
            yield
        finally:
            if worker is not None:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            if app.state.proxy is not None and hasattr(app.state.proxy, "close"):
                await app.state.proxy.close()

    app = FastAPI(title="LLMrouter", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.registry = model_registry
    app.state.router = app_router
    app.state.proxy = proxy
    app.state.collector = collector
    app.state.feedback_loop = feedback_loop
    app.state.api_key = api_key
    app.state.precog_publisher = precog_publisher
    app.state.precog_project = precog_project

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "models": len(app.state.registry.models),
            "providers": sorted(
                provider.value for provider in getattr(app.state.proxy, "providers", [])
            )
            if app.state.proxy is not None
            else [],
            "evaluator": app.state.feedback_loop is not None,
            "openai_compatible": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models",
                "routing_roles": _routing_roles(app.state.registry),
            },
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, object]:
        _require_api_key(request, app.state.api_key)
        return {
            "object": "list",
            "data": [_model_payload(model) for model in app.state.registry.all()],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionPayload,
        request: Request,
    ) -> Any:
        _require_api_key(request, app.state.api_key)
        if app.state.proxy is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Provider proxy is not configured",
            )

        chat_request = _to_chat_request(payload)

        # Debug: log incoming request
        _logger.debug(
            "POST /v1/chat/completions | model=%s, messages=%d, prompt_len=%d, stream=%s",
            payload.model,
            len(payload.messages),
            len(chat_request.prompt_text),
            payload.stream,
        )

        # Streaming path — SSE response for clients like Cline
        if payload.stream:
            return await _stream_response(
                request=request,
                chat_request=chat_request,
                payload=payload,
                proxy=app.state.proxy,
                app_router=app.state.router,
                collector=app.state.collector,
                precog_publisher=app.state.precog_publisher,
                precog_project=app.state.precog_project,
            )

        started = time.perf_counter()
        request_id = _request_id(request)
        constraints = _routing_constraints(payload)
        decision = await app.state.router.route(chat_request, constraints)

        # Debug: log routing decision
        _logger.debug(
            "Routing decision: primary=%s | score=%.2f tier=%s | fallbacks=%s",
            decision.primary.name,
            decision.score,
            decision.tier.name,
            [m.name for m in decision.fallbacks] or "none",
        )
        _logger.debug("Reason: %s", decision.reason)

        try:
            response = await app.state.proxy.chat_completion(chat_request, decision)
        except ProviderError as exc:
            _logger.warning("Provider error: %s (status=%d)", exc, exc.status_code)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        _log_chat_access(
            request=request,
            requested_model=payload.model or "auto",
            selected_model=decision.primary,
            status_code=status.HTTP_200_OK,
            stream=False,
        )

        # Debug: log response summary
        _logger.debug(
            "Response: %d tokens (prompt=%d, completion=%d) in %.0fms",
            response.usage.total_tokens,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            latency_ms,
        )
        _record_observation(
            collector=app.state.collector,
            chat_request=chat_request,
            response_payload=response.choices,
            model=decision.primary.name,
            selected_model=decision.primary,
            usage=response.usage,
            latency_ms=latency_ms,
            scorer_score=decision.score,
            scorer_tier=decision.tier.value,
            request_id=request_id,
            payload=payload,
            precog_publisher=app.state.precog_publisher,
            precog_project=app.state.precog_project,
        )
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created or int(time.time()),
            "model": response.model,
            "choices": response.choices,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "llmrouter": {
                "request_id": request_id,
                "selected_model": decision.primary.name,
                "provider": decision.primary.provider.value,
                "provider_model": decision.primary.provider_model_name,
                "score": decision.score,
                "tier": decision.tier.value,
                "reason": decision.reason,
            },
        }

    @app.post("/v1/llmrouter/feedback")
    async def llmrouter_feedback(
        payload: LLMrouterFeedbackPayload,
        request: Request,
    ) -> dict[str, object]:
        """Forward caller feedback to PRecog for a previously returned request_id."""
        _require_api_key(request, app.state.api_key)
        if app.state.precog_publisher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PRecog publisher is not configured",
            )
        app.state.precog_publisher.update_observation(payload.request_id, payload.outcome)
        return {"status": "accepted", "request_id": payload.request_id}

    @app.post("/admin/evaluator/run-cycle")
    async def run_evaluator_cycle(request: Request, limit: int = 50) -> dict[str, object]:
        _require_api_key(request, app.state.api_key)
        if app.state.feedback_loop is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feedback loop is not configured",
            )
        report = await app.state.feedback_loop.run_cycle(limit=limit)
        return {
            "evaluated": report.evaluated,
            "optimal": report.optimal,
            "correct": report.correct,
            "overkill": report.overkill,
            "underkill": report.underkill,
        }

    return app


async def _stream_response(
    *,
    request: Request,
    chat_request: ChatRequest,
    payload: ChatCompletionPayload,
    proxy: ProviderProxy,
    app_router: MultiModelRouter,
    collector: ObservationCollector | None,
    precog_publisher: Any | None = None,
    precog_project: str = "llmrouter",
) -> StreamingResponse:
    """Build a Server-Sent Events streaming response for chat completions.

    Routes the request through the multi-model router, then streams chunks
    from the selected provider proxy in OpenAI SSE format.
    """
    constraints = _routing_constraints(payload)
    decision = await app_router.route(chat_request, constraints)
    selected_model = decision.primary
    started = time.perf_counter()
    request_id = _request_id(request)

    _log_chat_access(
        request=request,
        requested_model=payload.model or "auto",
        selected_model=selected_model,
        status_code=status.HTTP_200_OK,
        stream=True,
    )

    # Debug: log routing decision for streaming
    _logger.debug(
        "Stream routing: primary=%s | score=%.2f tier=%s | fallbacks=%s",
        selected_model.name,
        decision.score,
        decision.tier.name,
        [m.name for m in decision.fallbacks] or "none",
    )
    _logger.debug("Reason: %s", decision.reason)

    async def event_generator() -> AsyncIterator[str]:
        collected_content: list[str] = []
        saw_output = False
        try:
            async for chunk in proxy.stream_chat_completion(chat_request, decision):
                normalized_chunk = _normalize_stream_chunk(chunk, selected_model.name)
                if normalized_chunk is None:
                    continue
                saw_output = saw_output or _chunk_has_assistant_output(normalized_chunk)
                # Forward a normalized OpenAI-compatible chunk to the client.
                yield f"data: {json.dumps(normalized_chunk)}\n\n"
                # Accumulate content for observation recording
                _extract_delta_text(normalized_chunk, collected_content)
            if not saw_output:
                _logger.warning(
                    "Provider stream completed without assistant content or tool calls: "
                    "selected=%s provider=%s provider_model=%s",
                    selected_model.name,
                    selected_model.provider.value,
                    selected_model.provider_model_name,
                )
        except ProviderError as exc:
            error_payload = {"error": {"message": str(exc), "type": "provider_error"}}
            yield f"data: {json.dumps(error_payload)}\n\n"
            return
        finally:
            yield "data: [DONE]\n\n"
            # Record observation (best-effort)
            latency_ms = (time.perf_counter() - started) * 1000
            if collector is not None:
                response_text = "".join(collected_content)
                # Approximate token count for observation
                approx_tokens = max(len(response_text) // 4, 1)
                usage = Usage(
                    prompt_tokens=len(chat_request.prompt_text) // 4,
                    completion_tokens=approx_tokens,
                    total_tokens=(len(chat_request.prompt_text) // 4) + approx_tokens,
                )
                _record_observation(
                    collector=collector,
                    chat_request=chat_request,
                    response_payload=[{"message": {"content": response_text}}],
                    model=selected_model.name,
                    selected_model=selected_model,
                    usage=usage,
                    latency_ms=latency_ms,
                    scorer_score=decision.score,
                    scorer_tier=decision.tier.value,
                    request_id=request_id,
                    payload=payload,
                    precog_publisher=precog_publisher,
                    precog_project=precog_project,
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-LLMrouter-Request-Id": request_id,
        },
    )


def _log_chat_access(
    *,
    request: Request,
    requested_model: str,
    selected_model: ModelInfo,
    status_code: int,
    stream: bool,
) -> None:
    client = request.client
    client_addr = f"{client.host}:{client.port}" if client else "-"
    _logger.info(
        '%s - "%s %s HTTP/%s" %d %s requested_model=%s selected_model=%s '
        "provider=%s provider_model=%s stream=%s",
        client_addr,
        request.method,
        request.url.path,
        request.scope.get("http_version", "1.1"),
        status_code,
        "OK" if status_code == status.HTTP_200_OK else "",
        requested_model,
        selected_model.name,
        selected_model.provider.value,
        selected_model.provider_model_name,
        stream,
    )


def _normalize_stream_chunk(chunk: dict[str, Any], model: str) -> dict[str, Any] | None:
    """Normalize provider SSE chunks to the OpenAI chat.completion.chunk shape."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    normalized_choices: list[dict[str, Any]] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        message = choice.get("message")
        if not isinstance(delta, dict):
            delta = {}
        if isinstance(message, dict):
            for key in ("role", "content", "tool_calls"):
                if key in message and key not in delta:
                    delta[key] = message[key]
        normalized_choices.append(
            {
                "index": int(choice.get("index", index) or 0),
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }
        )

    if not normalized_choices:
        return None
    return {
        "id": str(chunk.get("id") or f"chatcmpl-{int(time.time() * 1000)}"),
        "object": str(chunk.get("object") or "chat.completion.chunk"),
        "created": int(chunk.get("created") or int(time.time())),
        "model": str(chunk.get("model") or model),
        "choices": normalized_choices,
    }


def _chunk_has_assistant_output(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        tool_calls = delta.get("tool_calls")
        if isinstance(content, str) and content:
            return True
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def _extract_delta_text(chunk: dict[str, Any], accumulator: list[str]) -> None:
    """Extract delta text content from an SSE chunk for observation recording."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            accumulator.append(content)
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            accumulator.append(content)


def _to_chat_request(payload: ChatCompletionPayload) -> ChatRequest:
    # Merge explicit pass-through fields into extra
    extra = dict(payload.extra)
    if payload.tools is not None:
        extra["tools"] = payload.tools
    if payload.tool_choice is not None:
        extra["tool_choice"] = payload.tool_choice
    if payload.response_format is not None:
        extra["response_format"] = payload.response_format
    if payload.seed is not None:
        extra["seed"] = payload.seed
    if payload.frequency_penalty is not None:
        extra["frequency_penalty"] = payload.frequency_penalty
    if payload.presence_penalty is not None:
        extra["presence_penalty"] = payload.presence_penalty
    if payload.n is not None:
        extra["n"] = payload.n
    if payload.logit_bias is not None:
        extra["logit_bias"] = payload.logit_bias
    if payload.user is not None:
        extra["user"] = payload.user
    if payload.task_role is not None:
        extra["task_role"] = payload.task_role
    if payload.metadata is not None:
        extra["metadata"] = payload.metadata
    if payload.llmrouter is not None:
        extra["llmrouter"] = payload.llmrouter

    def _flatten_content(content: str | list[dict[str, Any]]) -> str:
        """Flatten content array to a single string for provider compatibility."""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)

    return ChatRequest(
        model=payload.model,
        messages=[
            ChatMessage(
                role=message.role,
                content=_flatten_content(message.content),
                name=message.name,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
            )
            for message in payload.messages
        ],
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
        top_p=payload.top_p,
        stop=payload.stop,
        extra=extra,
    )


def _model_payload(model: ModelInfo) -> dict[str, object]:
    return {
        "id": model.name,
        "object": "model",
        "owned_by": model.provider.value,
        "llmrouter": {
            "tier": model.tier.value,
            "capabilities": sorted(model.capabilities),
            "context_window": model.context_window,
            "api_base": model.api_base,
            "description": model.description,
        },
    }


def _record_observation(
    *,
    collector: ObservationCollector | None,
    chat_request: ChatRequest,
    response_payload: list[dict[str, Any]],
    model: str,
    selected_model: ModelInfo,
    usage: Usage,
    latency_ms: float,
    scorer_score: float,
    scorer_tier: int,
    request_id: str | None,
    payload: ChatCompletionPayload,
    precog_publisher: Any | None = None,
    precog_project: str = "llmrouter",
) -> None:
    if collector is None and precog_publisher is None:
        return
    response_text = "\n".join(_choice_text(choice) for choice in response_payload)
    cost_usd = _estimate_cost(selected_model, usage)
    metadata = {
        "provider": selected_model.provider.value,
        "provider_model": selected_model.provider_model_name,
    }
    if request_id:
        metadata["request_id"] = request_id
    if collector is not None:
        collector.record(
            RoutingObservation(
                prompt=chat_request.prompt_text,
                chosen_model=model,
                response=response_text,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                scorer_score=scorer_score,
                scorer_tier=scorer_tier,
                metadata=metadata,
            )
        )
    if precog_publisher is not None and request_id:
        precog_publisher.record_observation(
            {
                "request_id": request_id,
                "project": _precog_project(payload, precog_project),
                "task_role": _task_role(payload),
                "prompt_hash": _prompt_hash(chat_request.prompt_text),
                "selected_model": model,
                "provider": selected_model.provider.value,
                "provider_model": selected_model.provider_model_name,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": cost_usd,
                "rag": _rag_metadata(payload),
            }
        )


def _choice_text(choice: dict[str, Any]) -> str:
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = choice.get("text")
    return str(text) if text is not None else ""


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or f"llmrouter-{uuid.uuid4().hex}"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _precog_project(payload: ChatCompletionPayload, default: str) -> str:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    project = router_options.get("project") or payload.extra.get("project") or default
    return str(project or default)


def _task_role(payload: ChatCompletionPayload) -> str:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    role = (
        payload.task_role
        or router_options.get("task_role")
        or router_options.get("role")
        or payload.extra.get("role")
        or payload.extra.get("task_role")
        or ""
    )
    return str(role)


def _rag_metadata(payload: ChatCompletionPayload) -> dict[str, Any]:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    rag = router_options.get("rag")
    if not isinstance(rag, dict):
        return {"used": False, "collection": None, "top_k": 0, "context_tokens": 0}
    used = bool(rag.get("used"))
    return {
        "used": used,
        "collection": rag.get("collection") if used else None,
        "top_k": int(rag.get("top_k") or 0) if used else 0,
        "context_tokens": int(rag.get("context_tokens") or 0) if used else 0,
    }


def _routing_constraints(payload: ChatCompletionPayload) -> RoutingConstraints:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    role = (
        payload.task_role
        or router_options.get("task_role")
        or router_options.get("role")
        or payload.extra.get("role")
        or payload.extra.get("task_role")
    )
    if not isinstance(role, str) or not role:
        return RoutingConstraints()
    return RoutingConstraints(required_capabilities=frozenset({role}))


def _routing_roles(registry: ModelRegistry) -> list[str]:
    roles: set[str] = set()
    for model in registry.all():
        roles.update(model.capabilities)
    return sorted(roles)


def _estimate_cost(model: ModelInfo, usage: Usage) -> float:
    input_cost = (usage.prompt_tokens / 1000) * model.cost_per_1k_input
    output_cost = (usage.completion_tokens / 1000) * model.cost_per_1k_output
    return input_cost + output_cost


async def _run_feedback_worker(feedback_loop: FeedbackLoop, interval_seconds: int) -> None:
    interval = max(interval_seconds, 1)
    while True:
        await asyncio.sleep(interval)
        with contextlib.suppress(Exception):
            await feedback_loop.run_cycle()


def _require_api_key(request: Request, configured_api_key: str | None) -> None:
    if not configured_api_key:
        return

    x_api_key = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    bearer = authorization.removeprefix("Bearer ").strip()
    if x_api_key == configured_api_key or bearer == configured_api_key:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
