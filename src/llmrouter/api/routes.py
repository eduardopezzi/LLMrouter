"""FastAPI routes for the OpenAI-compatible gateway."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import difflib
import hashlib
import json
import re
import struct
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llmrouter.benchmark_scheduler import BenchmarkRefreshScheduler
from llmrouter.config import get_settings
from llmrouter.core.budget import (
    DEFAULT_PROJECT_ID,
    DEFAULT_USER_ID,
    BudgetLimits,
    BudgetManager,
)
from llmrouter.core.budget import estimate_cost as budget_estimate_cost
from llmrouter.core.cache import CacheManager
from llmrouter.core.health import ModelHealthTracker
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter, NoModelsAvailableError
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.semantic_cache import OllamaJudge, SemanticCache
from llmrouter.core.stats import MetricsCollector
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingConstraints,
    RoutingStrategy,
    Usage,
)
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.feedback import FeedbackLoop
from llmrouter.evaluator.types import RoutingObservation
from llmrouter.logging_config import get_logger
from llmrouter.memory import MemoryEntry, MemoryStore, render_memory_context
from llmrouter.providers.base import BaseProvider, ProviderError

_logger = get_logger("llmrouter.api")
_OBSERVATION_ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,255}$")
_PROJECT_ID_HEADER = "x-project-id"
_TASK_ROLE_HEADER = "x-task-role"


class ChatMessagePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessagePayload]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    top_p: float | None = 1.0
    stop: str | list[str] | None = None
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


class EmbeddingPayload(BaseModel):
    """OpenAI-compatible input for ``POST /v1/embeddings``."""

    model: str = Field(min_length=1, max_length=256)
    input: str | list[str]
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: int | None = Field(default=None, gt=0)
    user: str | None = None
    truncate: bool = True


class LLMrouterFeedbackPayload(BaseModel):
    """Post-execution feedback for an LLMrouter request."""

    request_id: str = Field(min_length=1, max_length=128)
    outcome: dict[str, Any] = Field(default_factory=dict)


class SemanticInspectPayload(BaseModel):
    """Prompt payload for scorer inspection without provider calls."""

    prompt: str = ""
    model: str | None = None


class CacheVerifyPayload(BaseModel):
    """Body for POST /v1/llmrouter/cache/verify — P-CHR judge run."""

    sample_size: int | None = Field(default=None, gt=0)


class BudgetLimitsPayload(BaseModel):
    """Body for POST /v1/llmrouter/budgets — set or replace tenant limits."""

    project_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=128)
    daily_limit_usd: float | None = None
    monthly_limit_usd: float | None = None
    mode: str = Field(default="soft", pattern="^(soft|hard)$")


class RagQueryPayload(BaseModel):
    """Body for POST /v1/llmrouter/rag/query — Dify-compatible retrieval."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(min_length=1, max_length=8192)
    dataset_id: str | None = Field(default=None, min_length=1, max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


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
    memory_store: MemoryStore | None = None,
    health_tracker: ModelHealthTracker | None = None,
    metrics_collector: MetricsCollector | None = None,
    cache_manager: CacheManager | None = None,
    semantic_cache: SemanticCache | None = None,
    budget_manager: BudgetManager | None = None,
    benchmark_scheduler: BenchmarkRefreshScheduler | None = None,
    ragflow_client: Any | None = None,
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
        benchmark_worker: asyncio.Task[None] | None = None
        if feedback_loop is not None and evaluator_interval_seconds:
            worker = asyncio.create_task(
                _run_feedback_worker(feedback_loop, evaluator_interval_seconds)
            )
        if benchmark_scheduler is not None:
            benchmark_worker = asyncio.create_task(benchmark_scheduler.run())
        try:
            yield
        finally:
            if worker is not None:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            if benchmark_worker is not None:
                benchmark_worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await benchmark_worker
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
    app.state.memory_store = memory_store
    app.state.health_tracker = health_tracker
    app.state.metrics_collector = metrics_collector
    app.state.cache_manager = cache_manager
    app.state.semantic_cache = semantic_cache
    app.state.budget_manager = budget_manager
    app.state.benchmark_scheduler = benchmark_scheduler
    app.state.ragflow_client = ragflow_client

    @app.get("/health/models")
    async def health_models(request: Request) -> dict[str, object]:
        _require_api_key(request, app.state.api_key)
        tracker: ModelHealthTracker | None = getattr(app.state, "health_tracker", None)
        if tracker is None:
            return {"models": []}
        return {
            "window_minutes": tracker.window_minutes,
            "models": [h.to_dict() for h in await tracker.list_health()],
        }

    @app.get("/health/models/{model_name}")
    async def health_model_detail(model_name: str, request: Request) -> dict[str, object]:
        _require_api_key(request, app.state.api_key)
        tracker: ModelHealthTracker | None = getattr(app.state, "health_tracker", None)
        if tracker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Health tracker is not configured",
            )
        score = await tracker.health_score(model_name)
        health = await tracker.get_health(model_name)
        return {
            "model": model_name,
            "score": score.to_dict(),
            "health": health.to_dict(),
        }

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
            "memory": app.state.memory_store is not None,
            "health_tracker": app.state.health_tracker is not None,
            "openai_compatible": {
                "chat_completions": "/v1/chat/completions",
                "embeddings": "/v1/embeddings",
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

    @app.post("/v1/embeddings")
    async def create_embeddings(
        payload: EmbeddingPayload,
        request: Request,
    ) -> dict[str, object]:
        """Generate OpenAI-compatible embeddings through the configured Ollama provider."""
        _require_api_key(request, app.state.api_key)

        raw_model = payload.model.strip()
        if not raw_model:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="model must not be blank",
            )
        if raw_model in {"auto", "ollama/auto"}:
            model_name = get_settings().semantic.model_name
        elif "/" in raw_model and raw_model.split("/", maxsplit=1)[0] in {
            "openai",
            "zai",
            "gemini",
            "deepseek",
        }:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Embeddings are currently supported only for Ollama models",
            )
        elif raw_model.startswith("ollama/"):
            _, model_name = raw_model.split("/", maxsplit=1)
        else:
            model_name = raw_model
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="model must not be blank",
            )

        inputs = [payload.input] if isinstance(payload.input, str) else payload.input
        if not inputs or any(not text.strip() for text in inputs):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="input must contain non-empty text",
            )

        proxy = getattr(app.state, "proxy", None)
        providers = getattr(proxy, "_providers", {})
        ollama = providers.get(Provider.OLLAMA) if isinstance(providers, dict) else None
        if ollama is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama embedding provider is not configured",
            )

        try:
            vectors, prompt_tokens = await ollama.embeddings(
                model=model_name,
                inputs=inputs,
                dimensions=payload.dimensions,
                truncate=payload.truncate,
            )
        except ProviderError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        data: list[dict[str, object]] = []
        for index, vector in enumerate(vectors):
            embedding: list[float] | str = vector
            if payload.encoding_format == "base64":
                packed = struct.pack(f"<{len(vector)}f", *vector)
                embedding = base64.b64encode(packed).decode("ascii")
            data.append({"object": "embedding", "index": index, "embedding": embedding})

        return {
            "object": "list",
            "data": data,
            "model": payload.model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

    @app.get("/v1/llmrouter/rollout")
    async def get_rollout_status(request: Request) -> dict[str, object]:
        """Return rollout percentages for all models in the catalog."""
        _require_api_key(request, app.state.api_key)
        return {
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider.value,
                    "rollout_percentage": m.rollout_percentage,
                }
                for m in app.state.registry.all()
            ]
        }

    @app.post("/v1/llmrouter/rollout/{model_name:path}")
    async def set_rollout_percentage(
        model_name: str,
        percentage: float,
        request: Request,
    ) -> dict[str, object]:
        """Update rollout percentage for a model at runtime (hot-reload registry).

        Persists the new value to the YAML catalog file and replaces the live
        registry used by the router, without requiring a server restart.
        """
        _require_api_key(request, app.state.api_key)
        if not 0.0 <= percentage <= 100.0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="percentage must be between 0 and 100",
            )

        from llmrouter.cli_panel import set_model_rollout_percentage
        from llmrouter.config import get_settings
        from llmrouter.core.registry import load_model_registry

        settings = get_settings()
        models_file = settings.models_file

        try:
            set_model_rollout_percentage(models_file, model_name, percentage)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        new_registry = load_model_registry(models_file)
        app.state.router.replace_registry(new_registry)
        app.state.registry = new_registry

        updated = new_registry.get(model_name)
        new_pct = updated.rollout_percentage if updated else percentage

        _logger.info(
            "routing_decision.rollout_updated model=%s percentage=%.1f",
            model_name,
            new_pct,
        )

        return {
            "model": model_name,
            "rollout_percentage": new_pct,
            "applied": True,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionPayload,
        request: Request,
    ) -> Any:
        _require_api_key(request, app.state.api_key)
        resource_policy = None
        raw_policy = request.headers.get("X-Resource-Policy")
        if raw_policy:
            from llmrouter.resource_policy import PolicyValidationError, parse_resource_policy

            try:
                resource_policy = parse_resource_policy(raw_policy)
            except PolicyValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"invalid X-Resource-Policy: {exc}",
                ) from exc
        if app.state.proxy is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Provider proxy is not configured",
            )

        payload = _with_observation_identity(payload, request)
        chat_request = _with_client_identity(_to_chat_request(payload), request)
        policy_clamped: int | None = None
        if resource_policy is not None:
            requested_max = chat_request.max_tokens
            if requested_max is None or requested_max > resource_policy.max_output_tokens:
                chat_request = replace(
                    chat_request,
                    max_tokens=resource_policy.max_output_tokens,
                )
                policy_clamped = requested_max if requested_max is not None else 0
            chat_request = replace(chat_request, resource_policy=resource_policy)
        # Budget pre-flight (B2): real pre-call cost estimation is infeasible
        # before routing selects a model, so the check runs with an estimated
        # cost of 0.0 and relies on post-response record_usage to accumulate
        # actual spend.  A hard-limit breach denies the request (402); a soft
        # breach is surfaced to the client via the X-Budget-Warning header.
        budget_manager = getattr(app.state, "budget_manager", None)
        budget_project, budget_user = _budget_tenant(request)
        budget_warning: str | None = None
        if budget_manager is not None:
            budget_decision = await budget_manager.check(budget_project, budget_user, 0.0)
            if not budget_decision.allowed:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=budget_decision.reason,
                )
            budget_warning = budget_decision.warning
            chat_request = replace(
                chat_request,
                budget_remaining_usd=budget_decision.remaining_usd,
            )
        prompt_directives = _chat_request_directives(chat_request)
        prompt_directives = _resolve_prompt_directives(
            prompt_directives,
            registry=model_registry,
            project_candidates=_project_candidates(
                prompt=chat_request.prompt_text,
                default=_memory_default_project(
                    app.state.memory_store,
                    app.state.precog_project,
                ),
                precog_project=app.state.precog_project,
            ),
        )
        chat_request = _with_prompt_directives(chat_request, payload, prompt_directives)
        original_chat_request = chat_request
        memory_project = _memory_project(
            payload,
            request,
            default=_memory_default_project(app.state.memory_store, app.state.precog_project),
            prompt=chat_request.prompt_text,
            directives=prompt_directives,
        )
        memory_repository = _precog_repository(payload)
        memory_project = _memory_scope_project(
            app.state.memory_store,
            project=memory_project,
            repository=memory_repository,
        )
        memory_entries = _retrieve_memory(
            app.state.memory_store,
            project=memory_project,
            repository=memory_repository,
            chat_request=chat_request,
            payload=payload,
        )
        chat_request = _with_memory_context(
            chat_request,
            memory_entries,
            memory_store=app.state.memory_store,
        )

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
                memory_store=app.state.memory_store,
                memory_project=memory_project,
                memory_repository=memory_repository,
                original_chat_request=original_chat_request,
                memory_entries=memory_entries,
                health_tracker=app.state.health_tracker,
                budget_manager=budget_manager,
                budget_project=budget_project,
                budget_user=budget_user,
                budget_warning=budget_warning,
            )

        started = time.perf_counter()
        request_id = _request_id(request)
        constraints = _routing_constraints(payload, prompt_directives)
        try:
            decision = await app.state.router.route(original_chat_request, constraints)
        except NoModelsAvailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

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

        # Log health score for the selected model
        await _log_selected_model_health(
            app.state.health_tracker,
            decision.primary.name,
            latency_ms,
        )

        # Debug: log response summary
        _logger.debug(
            "Response: %d tokens (prompt=%d, completion=%d, cache=%s) in %.0fms | model=%s tier=%s",
            response.usage.total_tokens,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            _cache_usage_label(response.usage),
            latency_ms,
            decision.primary.name,
            decision.tier.name if hasattr(decision, "tier") else "?",
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
            routing_strategy=app.state.router.routing_strategy.value,
            precog_publisher=app.state.precog_publisher,
            precog_project=app.state.precog_project,
            memory_entries=memory_entries,
        )
        _record_memory(
            app.state.memory_store,
            project=memory_project,
            chat_request=original_chat_request,
            response_payload=response.choices,
            selected_model=decision.primary,
            request_id=request_id,
            payload=payload,
            memory_entries=memory_entries,
            repository=memory_repository,
        )
        # Budget post-response recording (B2).  Best-effort only: budget is
        # observability + governance and must never break the chat response.
        response_headers: dict[str, str] = {}
        if resource_policy is not None:
            response_headers["X-Resource-Policy-Version"] = resource_policy.version
            max_policy_tokens = (
                resource_policy.max_context_tokens + resource_policy.max_output_tokens
            )
            response_headers["X-Budget-Remaining"] = str(
                max(0, max_policy_tokens - response.usage.total_tokens)
            )
            if policy_clamped is not None:
                response_headers["X-Resource-Policy-Clamped"] = "max_tokens"
        if budget_warning:
            response_headers["X-Budget-Warning"] = budget_warning
        if budget_manager is not None:
            await _record_budget_usage(
                budget_manager,
                project=budget_project,
                user=budget_user,
                model=decision.primary,
                usage=response.usage,
            )
        body: dict[str, Any] = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created or int(time.time()),
            "model": response.model,
            "choices": response.choices,
            "usage": _usage_payload(response.usage),
            "llmrouter": {
                "request_id": request_id,
                "selected_model": decision.primary.name,
                "provider": decision.primary.provider.value,
                "provider_model": decision.primary.provider_model_name,
                "score": decision.score,
                "tier": decision.tier.value,
                "reason": decision.reason,
                "memory": _memory_payload(memory_entries, memory_project),
            },
        }
        if not response_headers:
            return body
        return JSONResponse(content=body, headers=response_headers)

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

    @app.post("/v1/llmrouter/semantic/inspect")
    async def semantic_inspect(
        payload: SemanticInspectPayload,
        request: Request,
    ) -> dict[str, object]:
        """Inspect the configured scorer output for a prompt without provider calls."""
        _require_api_key(request, app.state.api_key)
        if app.state.router is None or not hasattr(app.state.router, "score_prompt"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Router scorer is not configured",
            )
        scoring = app.state.router.score_prompt(payload.prompt)
        return _semantic_inspect_payload(scoring)

    @app.get("/v1/llmrouter/stats")
    async def get_stats(request: Request) -> dict[str, object]:
        """Return consolidated operational metrics."""
        _require_api_key(request, app.state.api_key)
        collector: MetricsCollector | None = getattr(app.state, "metrics_collector", None)
        if collector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Metrics collector is not configured",
            )
        snapshot = await collector.snapshot()
        return {
            "uptime_seconds": round(collector.uptime_seconds, 2),
            **snapshot.to_dict(),
        }

    @app.get("/v1/llmrouter/cache/stats")
    async def get_cache_stats(request: Request) -> dict[str, object]:
        """Return cache hit/miss statistics."""
        _require_api_key(request, app.state.api_key)
        cache: CacheManager | None = getattr(app.state, "cache_manager", None)
        semantic: SemanticCache | None = getattr(app.state, "semantic_cache", None)
        if cache is None and semantic is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache manager is not configured",
            )
        payload: dict[str, object] = {}
        if cache is not None:
            stats = await cache.stats()
            payload.update(stats.to_dict())
        elif semantic is not None:
            # QA LOW-10 — keep the exact-cache keys present (zeros) so the
            # response shape matches the contract even without a cache manager.
            payload.setdefault("hits", 0)
            payload.setdefault("misses", 0)
        if semantic is not None:
            semantic_stats = getattr(semantic, "stats", None)
            if callable(semantic_stats):
                semantic_payload = semantic_stats()
                if isinstance(semantic_payload, dict):
                    payload.update(semantic_payload)
            payload["stream"] = semantic.stream_stats()
        return payload

    @app.get("/v1/llmrouter/rag/health")
    async def get_ragflow_health(request: Request) -> dict[str, object]:
        """Report RAGFlow client state (E5-B1)."""
        _require_api_key(request, app.state.api_key)
        ragflow = getattr(app.state, "ragflow_client", None)
        if ragflow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="RAGFlow coupling is disabled (ragflow.enabled=False)",
            )
        stats: dict[str, object] = dict(ragflow.stats)
        return stats

    @app.post("/v1/llmrouter/rag/query")
    async def rag_query(
        payload: RagQueryPayload, request: Request
    ) -> dict[str, object]:
        """Proxy a retrieval call to RAGFlow (B1 — Cenário Lite).

        Payload is Dify-compatible: ``knowledge_id`` + ``query``. The
        response carries either ``records`` (200) or ``degraded: true``
        (HTTP 200, RAGFlow temporarily unavailable) — never 5xx, so callers
        can attach fallback behavior uniformly.
        """
        _require_api_key(request, app.state.api_key)
        ragflow = getattr(app.state, "ragflow_client", None)
        if ragflow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="RAGFlow coupling is disabled (ragflow.enabled=False)",
            )

        result = await ragflow.retrieval(
            query=payload.query,
            dataset_id=payload.dataset_id,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )

        from llmrouter.core.ragflow_client import RagflowUnavailable

        if isinstance(result, RagflowUnavailable):
            return {
                "degraded": True,
                "error": result.error,
                "retry_after_seconds": result.retry_after_seconds,
                "records": [],
                "query": payload.query,
                "dataset_id": payload.dataset_id,
            }

        return {
            "degraded": False,
            "records": [
                {
                    "content": r.content,
                    "score": r.score,
                    "title": r.title,
                    "metadata": r.metadata,
                }
                for r in result
            ],
            "query": payload.query,
            "dataset_id": payload.dataset_id,
        }
    @app.post("/v1/llmrouter/cache/verify")
    async def verify_cache_hits(
        request: Request,
        payload: CacheVerifyPayload | None = None,
    ) -> dict[str, object]:
        """Audit pending semantic-cache hit-log rows with the local LLM judge.

        Runs :meth:`SemanticCache.verify_pending` on up to ``sample_size``
        pending rows (default: ``settings.semantic_cache.verify_sample_size``)
        using an :class:`OllamaJudge` built from the verify-judge settings.
        Returns the ``{checked, ok, mismatch, error, buckets}`` summary.
        """
        _require_api_key(request, app.state.api_key)
        semantic: SemanticCache | None = getattr(app.state, "semantic_cache", None)
        if semantic is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Semantic cache is not configured",
            )
        settings = get_settings()
        body_sample = payload.sample_size if payload is not None else None
        sample_size = body_sample or settings.semantic_cache.verify_sample_size
        judge = OllamaJudge(
            base_url=settings.semantic_cache.verify_judge_base_url,
            model=settings.semantic_cache.verify_judge_model,
            timeout_seconds=settings.semantic_cache.verify_judge_timeout_seconds,
        )
        try:
            return await semantic.verify_pending(sample_size, judge)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as 500
            _logger.error("P-CHR verify run failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Verify run failed: {exc}",
            ) from exc

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

    @app.get("/v1/llmrouter/budgets/{project_id}")
    async def get_budget(
        project_id: str,
        request: Request,
        user_id: str = DEFAULT_USER_ID,
    ) -> dict[str, object]:
        """Return the current-period budget usage and active limits for a tenant."""
        _require_api_key(request, app.state.api_key)
        mgr: BudgetManager | None = getattr(app.state, "budget_manager", None)
        if mgr is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Budget manager is not configured",
            )
        usage = await mgr.get_usage(project_id, user_id)
        return {
            "project_id": usage.project_id,
            "user_id": usage.user_id,
            "daily_spent_usd": usage.daily_spent_usd,
            "monthly_spent_usd": usage.monthly_spent_usd,
            "period_day": usage.period_day,
            "period_month": usage.period_month,
            "daily_limit_usd": usage.daily_limit_usd,
            "monthly_limit_usd": usage.monthly_limit_usd,
            "mode": usage.mode,
        }

    @app.post("/v1/llmrouter/budgets")
    async def set_budget(
        payload: BudgetLimitsPayload,
        request: Request,
    ) -> dict[str, object]:
        """Set or replace the budget limits for a tenant."""
        _require_api_key(request, app.state.api_key)
        mgr = getattr(app.state, "budget_manager", None)
        if mgr is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Budget manager is not configured",
            )
        limits = BudgetLimits(
            daily_limit_usd=payload.daily_limit_usd,
            monthly_limit_usd=payload.monthly_limit_usd,
            mode=payload.mode,  # type: ignore[arg-type]
        )
        await mgr.set_limits(payload.project_id, payload.user_id, limits)
        return {
            "ok": True,
            "project_id": payload.project_id,
            "user_id": payload.user_id,
            "daily_limit_usd": payload.daily_limit_usd,
            "monthly_limit_usd": payload.monthly_limit_usd,
            "mode": payload.mode,
        }

    return app


_STREAM_PROBE_CIRCUIT: dict[str, list[float]] = {}
# QA MEDIUM-9 — cap on chunks retained for cache storage (memory guard;
# providers that never emit finish_reason stop accumulating at this bound).
_STREAM_STORE_MAX_CHUNKS = 2048


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
    memory_store: MemoryStore | None = None,
    memory_project: str = "default",
    memory_repository: str = "",
    original_chat_request: ChatRequest | None = None,
    memory_entries: list[MemoryEntry] | None = None,
    health_tracker: ModelHealthTracker | None = None,
    budget_manager: BudgetManager | None = None,
    budget_project: str = DEFAULT_PROJECT_ID,
    budget_user: str = DEFAULT_USER_ID,
    budget_warning: str | None = None,
    semantic_cache: SemanticCache | None = None,
    selected_provider: BaseProvider | None = None,
    probe_soft_circuit: dict[str, list[float]] | None = None,
) -> StreamingResponse:
    """Build a Server-Sent Events streaming response for chat completions.

    Budget enforcement is pre-flight only in the streaming path.  The
    ``X-Budget-Warning`` header (when applicable) is attached to the
    ``StreamingResponse`` constructor so it appears before the first SSE
    chunk.  Post-response ``record_usage`` is not attempted here: the
    final token count is only known after the stream completes, and
    wrapping the generator in an async ``finally`` block would still miss
    client-side aborts.  This honest limitation is documented for a
    future iteration that instruments the proxy's stream-end usage hook.

    The optional ``semantic_cache``, ``selected_provider`` and
    ``probe_soft_circuit`` parameters are injected by the route layer (or
    by tests).  When omitted, the function falls back to ``request.app.state``
    for ``semantic_cache`` and disables the probe circuit breaker (live only).
    """
    original_chat_request = original_chat_request or chat_request
    prompt_directives = _chat_request_directives(original_chat_request)
    constraints = _routing_constraints(payload, prompt_directives)
    try:
        decision = await app_router.route(original_chat_request, constraints)
    except NoModelsAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    selected_model = decision.primary
    started = time.perf_counter()
    request_id = _request_id(request)
    memory_entries = memory_entries or []

    # ROADMAP_TOKEN_OPTIMIZATION E2 — resolve replay dependencies.
    # QA CRITICAL-1: the route call-site never injected ``selected_provider``,
    # so the replay path was dormant in production.  Resolve the provider
    # from the proxy's registry (keyed by the selected model's provider kind);
    # fall back to ``app.state`` for tests that inject a custom stub.
    if semantic_cache is None:
        semantic_cache = getattr(request.app.state, "semantic_cache", None)
    if selected_provider is None:
        selected_provider = getattr(request.app.state, "selected_provider", None)
    if selected_provider is None:
        provider_registry = getattr(proxy, "_providers", None)
        if provider_registry is not None:
            selected_provider = provider_registry.get(selected_model.provider)
    # QA CRITICAL-2: a per-call dict could never trip across requests; keep
    # the circuit state at module level so consecutive requests cooperate.
    if probe_soft_circuit is None:
        probe_soft_circuit = _STREAM_PROBE_CIRCUIT

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

    # ROADMAP_TOKEN_OPTIMIZATION E2 / QA HIGH-5 — resolve the replay decision
    # EAGERLY, before building the StreamingResponse.  Under real ASGI the
    # headers are serialized when the response starts; a decision made inside
    # the body generator can never influence them.  Deciding here lets the
    # ``X-LLMrouter-Cache-Status: semantic_hit`` header be set on the
    # response constructor, visible to real clients.
    stream_headers: dict[str, str] = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-LLMrouter-Request-Id": request_id,
    }
    if budget_warning:
        stream_headers["X-Budget-Warning"] = budget_warning
    # ROADMAP_TOKEN_OPTIMIZATION E2 — advertise semantic cache replay status
    # via a header (never inside the SSE chunk to preserve OpenAI compatibility).
    if semantic_cache is not None and getattr(
        semantic_cache, "stream_cache_enabled", False
    ):
        stream_headers["X-LLMrouter-Stream-Cache"] = "enabled"

    replay_decision: tuple[list[dict[str, Any]], int] | None = None
    if (
        semantic_cache is not None
        and getattr(semantic_cache, "stream_cache_enabled", False)
        and selected_provider is not None
        and probe_soft_circuit is not None
    ):
        replay_decision = await _maybe_replay_stream(
            semantic_cache=semantic_cache,
            selected_provider=selected_provider,
            chat_request=chat_request,
            selected_model=selected_model,
            probe_soft_circuit=probe_soft_circuit,
        )

    if replay_decision is not None:
        stream_headers["X-LLMrouter-Cache-Status"] = "semantic_hit"

    async def event_generator() -> AsyncIterator[str]:
        collected_content: list[str] = []
        saw_output = False
        replay_active = False
        replay_bytes = 0
        # QA HIGH-4 — live-path bookkeeping: real normalized chunks plus a
        # completion signal observed from the provider itself.
        raw_chunks: list[dict[str, Any]] = []
        saw_finish_reason = False
        try:
            # ROADMAP_TOKEN_OPTIMIZATION E2 — replay the cached chunks.  The
            # decision (lookup + probe) was already made eagerly above; this
            # branch only iterates the resolved chunks.
            if replay_decision is not None:
                cached_chunks, completion_tokens = replay_decision
                replay_active = True
                for cached_chunk in cached_chunks:
                    normalized_chunk = _normalize_stream_chunk(
                        cached_chunk, selected_model.name
                    )
                    if normalized_chunk is None:
                        continue
                    saw_output = (
                        saw_output
                        or _chunk_has_assistant_output(normalized_chunk)
                    )
                    line = f"data: {json.dumps(normalized_chunk)}\n\n"
                    replay_bytes += len(line)
                    yield line
                    _extract_delta_text(normalized_chunk, collected_content)
                try:
                    if completion_tokens and semantic_cache is not None:
                        semantic_cache.bump_stream_counter(
                            "stream_tokens_saved_total", completion_tokens
                        )
                except Exception:  # pragma: no cover - never block the replay
                    pass
                yield "data: [DONE]\n\n"
                return
            async for chunk in proxy.stream_chat_completion(chat_request, decision):
                normalized_chunk = _normalize_stream_chunk(chunk, selected_model.name)
                if normalized_chunk is None:
                    continue
                saw_output = saw_output or _chunk_has_assistant_output(normalized_chunk)
                # QA HIGH-4/MEDIUM-9 — keep the provider's normalized chunks
                # (preserving id/created) and note whether a terminal
                # finish_reason actually arrived; the cache store refuses
                # truncated streams and replays the real chunk sequence
                # instead of a single synthetic envelope.
                if raw_chunks is not None and len(raw_chunks) < _STREAM_STORE_MAX_CHUNKS:
                    raw_chunks.append(normalized_chunk)
                for choice in normalized_chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        saw_finish_reason = True
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
            # Persist a validated stream response into the cache (best-effort).
            # QA HIGH-4: only when the provider itself signalled completion.
            if (
                semantic_cache is not None
                and getattr(semantic_cache, "stream_cache_enabled", False)
                and saw_output
                and saw_finish_reason
            ):
                await _store_stream_response_if_valid(
                    semantic_cache=semantic_cache,
                    chat_request=chat_request,
                    selected_model=selected_model,
                    collected_content=collected_content,
                    raw_chunks=raw_chunks,
                )
            yield "data: [DONE]\n\n"
            return
        except ProviderError as exc:
            error_payload = {"error": {"message": str(exc), "type": "provider_error"}}
            yield f"data: {json.dumps(error_payload)}\n\n"
            return
        except GeneratorExit:
            # Client disconnected — surface for the storage layer to skip.
            if semantic_cache is not None:
                try:
                    semantic_cache.bump_stream_counter("stream_replay_error_total", 1)
                except Exception:
                    pass
            return
        finally:
            # QA: no yield inside ``finally`` — yielding after GeneratorExit
            # raises RuntimeError (async generator ignored GeneratorExit).
            latency_ms = (time.perf_counter() - started) * 1000
            response_text = "".join(collected_content)
            # Approximate token count for observation and memory metadata.
            approx_tokens = max(len(response_text) // 4, 1)
            usage = Usage(
                prompt_tokens=len(chat_request.prompt_text) // 4,
                completion_tokens=approx_tokens,
                total_tokens=(len(chat_request.prompt_text) // 4) + approx_tokens,
            )
            if collector is not None:
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
                    routing_strategy=app_router.routing_strategy.value,
                    precog_publisher=precog_publisher,
                    precog_project=precog_project,
                    memory_entries=memory_entries,
                )
            _record_memory(
                memory_store,
                project=memory_project,
                chat_request=original_chat_request,
                response_payload=[{"message": {"content": response_text}}],
                selected_model=selected_model,
                request_id=request_id,
                payload=payload,
                memory_entries=memory_entries,
                repository=memory_repository,
            )
            await _log_selected_model_health(health_tracker, selected_model.name, latency_ms)
            if replay_active and semantic_cache is not None:
                try:
                    semantic_cache.bump_stream_counter(
                        "stream_replays_total", 1
                    )
                    semantic_cache.bump_stream_counter(
                        "stream_replay_bytes_served_total", replay_bytes
                    )
                except Exception:  # pragma: no cover
                    pass

    response = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=stream_headers,
    )
    return response


async def _maybe_replay_stream(
    *,
    semantic_cache: SemanticCache,
    selected_provider: BaseProvider,
    chat_request: ChatRequest,
    selected_model: Any,
    probe_soft_circuit: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], int] | None:
    """Run the k-token probe and replay cached chunks when the probe matches.

    Returns ``(cached_chunks, completion_tokens)`` on a successful replay or
    ``None`` if the replay path should be skipped (miss / probe diverge /
    NotImplementedError / soft-circuit open).  All counter bumps happen
    here so the generator can stay focused on yielding.
    """
    # Soft-circuit: skip the probe for this model if too many recent probes
    # failed within the configured window.  Empty dict => always probe.
    threshold = semantic_cache.stream_probe_soft_circuit_threshold
    window = semantic_cache.stream_probe_soft_circuit_seconds
    circuit = probe_soft_circuit.setdefault(selected_model.name, [])
    now = time.monotonic()
    circuit[:] = [t for t in circuit if now - t <= window]
    if len(circuit) >= threshold:
        # Open-circuit: bypass replay entirely, fall through to the live
        # path in the caller.  ``circuit`` is left untouched so consecutive
        # requests within the window keep skipping the probe.
        return None

    try:
        cached = await semantic_cache.lookup_stream_response(
            chat_request.prompt_text,
            model=selected_model.name,
            tier=int(selected_model.tier),
            temperature=chat_request.temperature or 0.0,
            top_p=chat_request.top_p or 1.0,
            max_tokens=chat_request.max_tokens,
        )
    except Exception:  # pragma: no cover - defensive: never block live path
        cached = None
    if cached is None:
        return None

    cached_chunks, completion_tokens, first_k = cached
    # Run the k-token probe against the live provider.
    # QA MEDIUM-7 — honour stream_probe_timeout_seconds: a hung provider must
    # not stall the stream; fall back to live on timeout.
    k = semantic_cache.stream_probe_k
    try:
        prefix = await asyncio.wait_for(
            selected_provider.first_tokens(chat_request, selected_model.name, k),
            timeout=semantic_cache.stream_probe_timeout_seconds,
        )
    except NotImplementedError:
        semantic_cache.bump_stream_counter("stream_probes_fail_total", 1)
        circuit.append(now)
        return None
    except Exception:
        semantic_cache.bump_stream_counter("stream_probes_fail_total", 1)
        circuit.append(now)
        return None

    if (prefix or "").strip() != (first_k or "").strip():
        semantic_cache.bump_stream_counter("stream_probes_fail_total", 1)
        circuit.append(now)
        return None

    # Successful replay — reset circuit, bump counters.
    circuit.clear()
    semantic_cache.bump_stream_counter("stream_probes_ok_total", 1)
    return cached_chunks, completion_tokens


async def _store_stream_response_if_valid(
    *,
    semantic_cache: SemanticCache,
    chat_request: ChatRequest,
    selected_model: Any,
    collected_content: list[str],
    raw_chunks: list[dict[str, Any]] | None,
) -> None:
    """Persist the live stream response into the cache when it ended cleanly.

    QA HIGH-4/MEDIUM-9 — the caller only reaches this point when the provider
    emitted a terminal ``finish_reason``.  We store the provider's real
    normalized chunks (preserving ``id``/``created``) so the replay is
    chunk-by-chunk instead of a single synthetic envelope.  A synthetic
    envelope is used only as a last-resort fallback when no chunks were
    captured, and it is never fabricated without the caller's completion
    guarantee.
    """
    response_text = "".join(collected_content)
    if not response_text:
        return
    approx_tokens = max(len(response_text) // 4, 1)
    usage = Usage(
        prompt_tokens=len(chat_request.prompt_text) // 4,
        completion_tokens=approx_tokens,
        total_tokens=(len(chat_request.prompt_text) // 4) + approx_tokens,
    )
    chunks_to_store: list[dict[str, Any]]
    if raw_chunks:
        chunks_to_store = raw_chunks
    else:  # pragma: no cover - defensive fallback
        chunks_to_store = [
            {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": selected_model.name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        ]
    try:
        await semantic_cache.store_stream_response(
            chunks_to_store,
            prompt=chat_request.prompt_text,
            usage=usage,
            model=selected_model.name,
            tier=int(selected_model.tier),
            temperature=chat_request.temperature or 0.0,
            top_p=chat_request.top_p or 1.0,
            max_tokens=chat_request.max_tokens,
            k=semantic_cache.stream_probe_k,
        )
    except Exception:  # pragma: no cover - never break the response
        pass


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
    extra = dict(payload.model_extra or {})
    extra.update(payload.extra)
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

    def _flatten_content(content: Any) -> str:
        """Flatten content array to a single string for provider compatibility."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)

    def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        return stop

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
        temperature=payload.temperature if payload.temperature is not None else 1.0,
        max_tokens=payload.max_tokens or payload.max_completion_tokens,
        stream=payload.stream,
        top_p=payload.top_p if payload.top_p is not None else 1.0,
        top_p_explicit=payload.top_p is not None and "top_p" in payload.model_fields_set,
        stop=_normalize_stop(payload.stop),
        extra=extra,
    )


def _with_client_identity(chat_request: ChatRequest, request: Request) -> ChatRequest:
    """Attach caller identity hints used for provider affinity and diagnostics."""
    client_ip = _client_ip(request)
    if not client_ip:
        return chat_request
    extra = dict(chat_request.extra)
    extra.setdefault("_llmrouter_client_ip", client_ip)
    extra.setdefault(
        "_llmrouter_client_id",
        extra.get("user") or request.headers.get("x-llmrouter-user") or client_ip,
    )
    return replace(chat_request, extra=extra)


def _with_observation_identity(
    payload: ChatCompletionPayload,
    request: Request,
) -> ChatCompletionPayload:
    """Use bounded proxy headers only when the request body omits telemetry identity."""
    extra = dict(payload.extra)
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    project = request.headers.get(_PROJECT_ID_HEADER, "").strip()
    if (
        project
        and _OBSERVATION_ID_RE.fullmatch(project)
        and "project" not in extra
        and not router_options.get("project")
    ):
        extra["project"] = project
    task_role = request.headers.get(_TASK_ROLE_HEADER, "").strip()
    updates: dict[str, Any] = {"extra": extra}
    if (
        task_role
        and _OBSERVATION_ID_RE.fullmatch(task_role)
        and payload.task_role is None
        and not router_options.get("task_role")
        and not router_options.get("role")
    ):
        updates["task_role"] = task_role
    return payload.model_copy(update=updates)


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client is not None:
        return request.client.host
    return ""


def _memory_default_project(
    memory_store: MemoryStore | None,
    fallback: str,
) -> str:
    if memory_store is None:
        return fallback
    return memory_store.config.default_project or fallback


def _memory_scope_project(
    memory_store: MemoryStore | None,
    *,
    project: str,
    repository: str,
) -> str:
    """Resolve the project namespace when repository provenance is absent."""
    if memory_store is None or repository:
        return project
    scope = str(getattr(memory_store.config, "no_repository_scope", "project")).lower()
    if scope == "global":
        return memory_store.config.default_project or project
    return project


def _memory_project(
    payload: ChatCompletionPayload,
    request: Request,
    *,
    default: str,
    prompt: str = "",
    directives: dict[str, str] | None = None,
) -> str:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    directives = directives or {}
    project = (
        request.headers.get("x-llmrouter-project")
        or router_options.get("project")
        or metadata.get("project")
        or payload.extra.get("project")
        or directives.get("project")
        or _infer_project_from_prompt(prompt)
        or default
    )
    result = str(project or default)
    _logger.debug(
        "Memory project resolved: project=%s sources=headers=%s router=%s metadata=%s "
        "extra=%s directive=%s inferred=%s",
        result,
        bool(request.headers.get("x-llmrouter-project")),
        router_options.get("project") is not None,
        metadata.get("project") is not None,
        payload.extra.get("project") is not None,
        directives.get("project") is not None,
        _infer_project_from_prompt(prompt) is not None,
    )
    return result


def _precog_repository(payload: ChatCompletionPayload) -> str:
    """Read optional repository provenance from supported request metadata."""
    for metadata in (payload.llmrouter, payload.metadata, payload.extra):
        if not isinstance(metadata, dict):
            continue
        repository = metadata.get("repository") or metadata.get("repo")
        if isinstance(repository, str) and repository.strip():
            return repository.strip()
    return ""


def _chat_request_directives(chat_request: ChatRequest) -> dict[str, str]:
    """Parse prompt directives from the leading lines of each message."""
    result: dict[str, str] = {}
    for message in chat_request.messages:
        result.update(_prompt_directives(message.content))
    if not result:
        result.update(_prompt_directives(chat_request.prompt_text))
    return result


def _prompt_directives(prompt: str | list[dict[str, Any]]) -> dict[str, str]:
    """Parse {{project:...}} style directives from leading prompt lines."""
    result: dict[str, str] = {}
    if isinstance(prompt, list):
        prompt = "\n".join(
            str(block.get("text") or block.get("content") or "")
            for block in prompt
            if isinstance(block, dict)
        )
    first_lines = "\n".join(prompt.splitlines()[:5])
    if not first_lines:
        return result
    aliases = {
        "p": "project",
        "project": "project",
        "t": "task_role",
        "task": "task_role",
        "task_role": "task_role",
        "role": "task_role",
        "m": "model",
        "model": "model",
        "preferred_model": "model",
    }
    directive_pattern = r"\{\{\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*([^{}]+?)\s*\}\}"
    for key, value in re.findall(directive_pattern, first_lines):
        normalized_key = aliases.get(key.strip().lower())
        if normalized_key is None:
            continue
        cleaned_value = value.strip().strip("\"'")
        if cleaned_value:
            result[normalized_key] = cleaned_value
    return result


def _resolve_prompt_directives(
    directives: dict[str, str],
    *,
    registry: ModelRegistry,
    project_candidates: list[str],
) -> dict[str, str]:
    if not directives:
        return directives
    resolved = dict(directives)
    if model := resolved.get("model"):
        model_match = _closest_model_name(model, registry)
        if model_match:
            if model_match != model:
                _logger.debug("Prompt directive model fuzzy matched: %s -> %s", model, model_match)
            resolved["model"] = model_match
        elif model.strip().casefold() not in {"auto", "default"}:
            _logger.debug("Ignoring unmatched prompt model directive: %s", model)
            resolved.pop("model", None)
    if project := resolved.get("project"):
        project_match = _closest_word(project, project_candidates, cutoff=0.6)
        if project_match and project_match != project:
            _logger.debug(
                "Prompt directive project fuzzy matched: %s -> %s",
                project,
                project_match,
            )
            resolved["project"] = project_match
    return resolved


def _closest_model_name(term: str, registry: ModelRegistry) -> str | None:
    normalized_term = term.strip().strip("<>[]{}()").strip()
    if not normalized_term or normalized_term.casefold() in {
        "model",
        "model_id",
        "model-id",
        "model name",
        "model_name",
        "placeholder",
    }:
        return None
    choices: dict[str, str] = {}
    for model in registry.models:
        choices[model.name] = model.name
        choices[model.provider_model_name] = model.name
        choices[model.name.removeprefix(f"{model.provider.value}/")] = model.name
    matched = _closest_word(normalized_term, list(choices), cutoff=0.7)
    return choices.get(matched or "")


def _closest_word(term: str, words: list[str], *, cutoff: float = 0.0) -> str | None:
    unique_words = [word for word in dict.fromkeys(words) if word]
    if not term or not unique_words:
        return None
    exact = {word.casefold(): word for word in unique_words}
    if term.casefold() in exact:
        return exact[term.casefold()]
    result = difflib.get_close_matches(term, unique_words, n=1, cutoff=cutoff)
    return result[0] if result else None


def _project_candidates(
    *,
    prompt: str,
    default: str,
    precog_project: str,
) -> list[str]:
    candidates = [default, precog_project]
    inferred = _infer_project_from_prompt(prompt)
    if inferred:
        candidates.append(inferred)
    for root in _project_roots():
        candidates.extend(_child_directory_names(root))
    return [candidate for candidate in dict.fromkeys(candidates) if candidate]


def _project_roots() -> list[Path]:
    home = Path.home()
    roots = [
        Path.cwd().parent,
        home / "github",
        home / "repos",
        home / "projects",
        home / "workspace",
        home / "workspaces",
    ]
    return [root for root in dict.fromkeys(roots) if root.exists() and root.is_dir()]


def _child_directory_names(root: Path, *, limit: int = 500) -> list[str]:
    names: list[str] = []
    try:
        for index, child in enumerate(root.iterdir()):
            if index >= limit:
                break
            if child.is_dir() and not child.name.startswith("."):
                names.append(child.name)
    except OSError:
        return []
    return names


def _with_prompt_directives(
    chat_request: ChatRequest,
    payload: ChatCompletionPayload,
    directives: dict[str, str],
) -> ChatRequest:
    if not directives:
        return chat_request
    model = directives.get("model")
    request_model = payload.model or ""
    next_model = chat_request.model
    if model and (not request_model or request_model == "auto"):
        next_model = None if model == "auto" else model
    extra = dict(chat_request.extra)
    extra["llmrouter_prompt_directives"] = directives
    return replace(chat_request, model=next_model, extra=extra)


def _infer_project_from_prompt(prompt: str) -> str | None:
    if not prompt:
        return None
    patterns = (
        r"Current Workspace Directory\s*\(([^)]+)\)",
        r"(?:workspace|project root|working directory|cwd)\s*[:=]\s*`?([^\n`]+)",
        r"/(?:github|repos|projects)/([A-Za-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match is None:
            continue
        raw = match.group(1).strip().strip("\"'")
        if "/" in raw:
            raw = raw.rstrip("/").rsplit("/", 1)[-1]
        project = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
        if project:
            return project
    return None


def _retrieve_memory(
    memory_store: MemoryStore | None,
    *,
    project: str,
    repository: str = "",
    chat_request: ChatRequest,
    payload: ChatCompletionPayload,
) -> list[MemoryEntry]:
    if memory_store is None:
        _logger.debug("Memory retrieval skipped: memory store not configured")
        return []
    if _memory_disabled(payload):
        _logger.debug(
            "Memory retrieval skipped: project=%s reason=memory_disabled via payload", project
        )
        return []
    if (
        not repository
        and str(getattr(memory_store.config, "no_repository_scope", "project")).lower()
        == "disabled"
    ):
        _logger.debug(
            "Memory retrieval skipped: project=%s reason=no_repository_scope_disabled",
            project,
        )
        return []
    query = chat_request.routing_prompt_text(max_chars=memory_store.config.query_max_chars)
    query_len = len(query)
    if repository:
        entries = memory_store.retrieve(project=project, query=query, repository=repository)
    else:
        entries = memory_store.retrieve(project=project, query=query)
    if entries:
        _logger.debug(
            "Memory retrieval: project=%s query_len=%d hits=%d ids=%s scores=%s",
            project,
            query_len,
            len(entries),
            [entry.id for entry in entries],
            [round(entry.score, 3) for entry in entries],
        )
    else:
        _logger.debug("Memory retrieval empty: project=%s query_len=%d", project, query_len)
    return entries


def _with_memory_context(
    chat_request: ChatRequest,
    memory_entries: list[MemoryEntry],
    *,
    memory_store: MemoryStore | None,
) -> ChatRequest:
    if memory_store is None or not memory_entries:
        return chat_request
    context = render_memory_context(
        memory_entries,
        max_chars=memory_store.config.max_context_chars,
    )
    if not context:
        _logger.debug(
            "Memory context empty after rendering: project=%s entries=%d",
            memory_store.config.default_project if memory_store else None,
            len(memory_entries),
        )
        return chat_request
    _logger.debug(
        "Memory context injected: project=%s entries=%d context_chars=%d ids=%s",
        memory_store.config.default_project if memory_store else "unknown",
        len(memory_entries),
        len(context),
        [entry.id for entry in memory_entries],
    )
    messages = [
        ChatMessage(role="system", content=context),
        *chat_request.messages,
    ]
    extra = dict(chat_request.extra)
    extra["llmrouter_memory"] = {
        "used": True,
        "ids": [entry.id for entry in memory_entries],
    }
    return replace(chat_request, messages=messages, extra=extra)


def _record_memory(
    memory_store: MemoryStore | None,
    *,
    project: str,
    chat_request: ChatRequest,
    response_payload: list[dict[str, Any]],
    selected_model: ModelInfo,
    request_id: str,
    payload: ChatCompletionPayload,
    memory_entries: list[MemoryEntry],
    repository: str = "",
) -> None:
    if memory_store is None or _memory_disabled(payload):
        return
    if (
        not repository
        and str(getattr(memory_store.config, "no_repository_scope", "project")).lower()
        == "disabled"
    ):
        return
    response_text = "\n".join(_choice_text(choice) for choice in response_payload).strip()
    metadata = {
        "request_id": request_id,
        "task_role": _task_role(payload, _chat_request_directives(chat_request)),
        "selected_model": selected_model.name,
        "provider": selected_model.provider.value,
        "provider_model": selected_model.provider_model_name,
        "retrieved_memory_ids": [entry.id for entry in memory_entries],
    }
    _logger.debug(
        "Memory recording attempt: project=%s prompt_len=%d response_len=%d model=%s "
        "retrieved_ids=%s",
        project,
        len(chat_request.prompt_text),
        len(response_text),
        selected_model.name,
        [entry.id for entry in memory_entries],
    )
    record_kwargs: dict[str, Any] = {
        "project": project,
        "prompt": chat_request.prompt_text,
        "response": response_text,
        "metadata": metadata,
    }
    if repository:
        record_kwargs["repository"] = repository
    recorded = memory_store.record_interaction(**record_kwargs)
    if recorded:
        _logger.debug(
            "Memory recorded successfully: project=%s model=%s", project, selected_model.name
        )
    else:
        _logger.debug(
            "Memory not recorded: project=%s model=%s reason=min_size_or_filters",
            project,
            selected_model.name,
        )


def _memory_disabled(payload: ChatCompletionPayload) -> bool:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    memory = router_options.get("memory")
    if isinstance(memory, dict):
        return memory.get("enabled") is False or memory.get("used") is False
    if isinstance(memory, bool):
        return not memory
    return False


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
            "benchmark_scores": dict(model.benchmark_scores),
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
    routing_strategy: str = "unknown",
    precog_publisher: Any | None = None,
    precog_project: str = "llmrouter",
    memory_entries: list[MemoryEntry] | None = None,
) -> None:
    if collector is None and precog_publisher is None:
        return
    response_text = "\n".join(_choice_text(choice) for choice in response_payload)
    cost_usd = _estimate_cost(selected_model, usage)
    metadata = {
        "provider": selected_model.provider.value,
        "provider_model": selected_model.provider_model_name,
        "routing_strategy": routing_strategy,
        "scorer_tier": str(scorer_tier),
        "rag_used": str(_rag_metadata(payload)["used"]).lower(),
        "memory_used": str(bool(memory_entries)).lower(),
    }
    rag = _rag_metadata(payload)
    metadata["rag_collection"] = str(rag["collection"] or "")
    metadata["rag_top_k"] = str(rag["top_k"])
    metadata["rag_context_tokens"] = str(rag["context_tokens"])
    if request_id:
        metadata["request_id"] = request_id
    if memory_entries:
        metadata["memory_ids"] = ",".join(str(entry.id) for entry in memory_entries)
    resource_policy = chat_request.resource_policy
    if resource_policy is not None:
        policy_id = getattr(resource_policy, "policy_id", None) or getattr(
            resource_policy, "id", None
        )
        if policy_id is not None:
            metadata["resource_policy_id"] = str(policy_id)
        policy_version = getattr(resource_policy, "version", None)
        if policy_version is not None:
            metadata["resource_policy_version"] = str(policy_version)
    if chat_request.budget_remaining_usd is not None:
        metadata["budget_remaining_usd"] = chat_request.budget_remaining_usd
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
                "cached_tokens": usage.cached_tokens,
                "cache_status": usage.cache_status,
                "cost_usd": cost_usd,
                "rag": _rag_metadata(payload),
                "memory": _memory_payload(
                    memory_entries or [],
                    _precog_project(payload, precog_project),
                ),
            }
        )


def _cache_usage_label(usage: Usage) -> str:
    """Render cache telemetry without representing an unknown value as zero."""
    if usage.cached_tokens is None:
        return "not_reported"
    return f"{usage.cache_status}:{usage.cached_tokens}"


def _usage_payload(usage: Usage) -> dict[str, Any]:
    """Build OpenAI-compatible usage data with optional cache provenance."""
    payload: dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cache_status": usage.cache_status,
    }
    if usage.cached_tokens is not None:
        payload["cached_tokens"] = usage.cached_tokens
        payload["prompt_tokens_details"] = {
            "cached_tokens": usage.cached_tokens,
        }
    return payload


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


def _task_role(
    payload: ChatCompletionPayload,
    directives: dict[str, str] | None = None,
) -> str:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    directives = directives or {}
    role = (
        payload.task_role
        or router_options.get("task_role")
        or router_options.get("role")
        or payload.extra.get("role")
        or payload.extra.get("task_role")
        or directives.get("task_role")
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


def _memory_payload(entries: list[MemoryEntry], project: str) -> dict[str, Any]:
    return {
        "used": bool(entries),
        "project": project,
        "top_k": len(entries),
        "ids": [entry.id for entry in entries],
    }


def _routing_constraints(
    payload: ChatCompletionPayload,
    directives: dict[str, str] | None = None,
) -> RoutingConstraints:
    router_options = payload.llmrouter if isinstance(payload.llmrouter, dict) else {}
    directives = directives or {}
    preferred_provider = _preferred_provider(payload, router_options)
    role = (
        payload.task_role
        or router_options.get("task_role")
        or router_options.get("role")
        or payload.extra.get("role")
        or payload.extra.get("task_role")
        or directives.get("task_role")
    )
    if not isinstance(role, str) or not role:
        return RoutingConstraints(preferred_provider=preferred_provider)
    return RoutingConstraints(
        required_capabilities=frozenset({role}),
        preferred_provider=preferred_provider,
    )


def _preferred_provider(
    payload: ChatCompletionPayload,
    router_options: dict[str, Any],
) -> Provider | None:
    raw = (
        router_options.get("provider")
        or payload.extra.get("provider")
        or payload.extra.get("preferred_provider")
    )
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return Provider(raw.strip().lower())
    except ValueError:
        _logger.debug("Ignoring unknown preferred provider: %s", raw)
        return None


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


def _semantic_inspect_payload(scoring: Any) -> dict[str, object]:
    signals = dict(getattr(scoring, "signals", {}) or {})
    confidence = signals.get("semantic_confidence", 0.0)
    try:
        semantic_confidence = float(confidence or 0.0)
    except (TypeError, ValueError):
        semantic_confidence = 0.0
    return {
        "score": getattr(scoring, "score", 0.0),
        "tier": getattr(getattr(scoring, "tier", None), "value", None),
        "complexity_level": signals.get("complexity_level", "unknown"),
        "task_type": signals.get("task_type", "general"),
        "semantic_role": signals.get("semantic_role", "none"),
        "semantic_confidence": semantic_confidence,
        "semantic_used": bool(signals.get("semantic_used", False)),
        "benchmark_top": signals.get("benchmark_top", "none"),
        "benchmark_affinities": signals.get("benchmark_affinities", {}),
        "benchmark_used": bool(signals.get("benchmark_used", False)),
        "signals": signals,
    }


async def _log_selected_model_health(
    tracker: ModelHealthTracker | None,
    model_name: str,
    latency_ms: float,
) -> None:
    """Log the composite health score for a selected model to the debug log.

    This gives visibility into real-time health metrics per-request,
    complementing the periodic HealthSummary from ModelHealthTracker.
    """
    if tracker is None:
        return
    try:
        score, health = await asyncio.gather(
            tracker.health_score(model_name),
            tracker.get_health(model_name),
        )
        _logger.debug(
            "HealthPerRequest model=%s score=%.4f latency=[p95=%.0fms p50=%.0fms req=%.0fms] "
            "error_rate=%.2f%% req_count=%d",
            model_name,
            score.score,
            health.p95_ms,
            health.p50_ms,
            latency_ms,
            health.error_rate * 100.0,
            health.request_count,
        )
    except Exception:
        _logger.debug("HealthPerRequest failed for %s (non-fatal)", model_name, exc_info=True)


def _budget_tenant(request: Request) -> tuple[str, str]:
    """Resolve the budget tenant from X-Project-ID / X-User-ID headers.

    Missing or empty headers fall back to the shared default tenant defined
    in :mod:`llmrouter.core.budget`.
    """
    project = request.headers.get("x-project-id", "").strip()
    user = request.headers.get("x-user-id", "").strip()
    return (
        project or DEFAULT_PROJECT_ID,
        user or DEFAULT_USER_ID,
    )


async def _record_budget_usage(
    budget_manager: BudgetManager,
    *,
    project: str,
    user: str,
    model: ModelInfo,
    usage: Usage,
) -> None:
    """Record post-response spend for a tenant.  Never raises.

    ``cost_known`` is ``False`` when the selected model has no price in the
    catalog (both per-1k rates are zero), which triggers the one-shot
    ``budget_zero_cost_usage`` warning inside the budget manager instead of
    silently polluting the spend accounting.
    """
    try:
        cost = budget_estimate_cost(
            model.cost_per_1k_input,
            model.cost_per_1k_output,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        cost_known = not (model.cost_per_1k_input == 0.0 and model.cost_per_1k_output == 0.0)
        await budget_manager.record_usage(project, user, cost, cost_known=cost_known)
        _logger.debug(
            "Budget usage recorded: project=%s user=%s model=%s cost=%.6f cost_known=%s",
            project,
            user,
            model.name,
            cost,
            cost_known,
        )
    except Exception:
        # Budget is observability + control; it must never break the chat.
        _logger.warning(
            "Budget usage recording failed (non-fatal): project=%s user=%s model=%s",
            project,
            user,
            model.name,
            exc_info=True,
        )
