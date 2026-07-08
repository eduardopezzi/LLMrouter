"""Runtime assembly from settings and model catalog."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI

from llmrouter.api.routes import create_app
from llmrouter.cli_panel import demote_model_priority
from llmrouter.config import ProviderConfig, Settings, get_settings
from llmrouter.core.cooldown import ProviderCooldownStore, is_quota_exhaustion_error
from llmrouter.core.health import (
    HealthBackend,
    HealthWeights,
    InMemoryHealthStore,
    ModelHealthTracker,
    ReviewQualitySource,
    SQLiteHealthStore,
)
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry, load_model_registry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer, ScorerWeights
from llmrouter.core.types import ModelInfo, Provider
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.feedback import FeedbackLoop
from llmrouter.evaluator.grader import RoutingDecisionGrader
from llmrouter.evaluator.judge import QualityJudge
from llmrouter.memory import MemoryConfig, PrecogMemoryConfig, PrecogMemoryStore, SQLiteMemoryStore
from llmrouter.precog import PrecogPublisher
from llmrouter.providers import (
    BaseProvider,
    DeepSeekProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    ZaiProvider,
)
from llmrouter.providers.base import ProviderError
from llmrouter.utils import resolve_api_key


def build_app(settings: Settings | None = None) -> FastAPI:
    """Create a fully wired FastAPI app from configuration."""
    resolved_settings = settings or get_settings()
    _ensure_runtime_logging(debug=resolved_settings.debug)
    registry = build_registry(resolved_settings.models_file)
    health_tracker = (
        _build_health_tracker(resolved_settings)
        if resolved_settings.health.enabled
        else None
    )
    provider_cooldowns = ProviderCooldownStore(
        default_seconds=resolved_settings.routing.quota_cooldown_seconds
    )
    router = MultiModelRouter(
        registry=registry,
        scorer=PromptScorer(_scorer_weights(resolved_settings.routing.scorer_weights)),
        strategy=resolved_settings.routing.strategy,
        fallback_count=resolved_settings.routing.fallback_count,
        provider_cost_order=resolved_settings.routing.provider_cost_order,
        health_tracker=health_tracker,
        rollout_config=resolved_settings.rollout,
        provider_cooldowns=provider_cooldowns,
        client_provider_affinity=resolved_settings.routing.client_provider_affinity,
    )
    app_holder: dict[str, FastAPI] = {}
    proxy_holder: dict[str, ProviderProxy] = {}
    proxy = ProviderProxy(
        build_providers(resolved_settings, registry),
        on_provider_error=_priority_demoter(
            resolved_settings.models_file,
            router,
            provider_cooldowns,
            app_holder,
            proxy_holder,
        ),
        health_tracker=health_tracker,
        provider_cooldowns=provider_cooldowns,
    )
    proxy_holder["proxy"] = proxy
    collector = (
        ObservationCollector(
            db_path=resolved_settings.evaluator.db_path,
            buffer_size=resolved_settings.evaluator.collection.buffer_size,
            sample_rate=resolved_settings.evaluator.collection.sample_rate,
        )
        if resolved_settings.evaluator.enabled
        else None
    )
    feedback_loop = (
        FeedbackLoop(
            collector=collector,
            judge=QualityJudge(
                base_url=resolved_settings.evaluator.ollama.base_url,
                api_key=resolve_api_key(resolved_settings.evaluator.ollama, "OLLAMA_API_KEY"),
                model=resolved_settings.evaluator.ollama.model,
                timeout=resolved_settings.evaluator.ollama.timeout,
                temperature=resolved_settings.evaluator.ollama.temperature,
            ),
            grader=RoutingDecisionGrader(),
            registry=registry,
        )
        if collector is not None
        else None
    )
    precog_publisher = (
        PrecogPublisher(
            base_url=resolved_settings.precog.base_url,
            api_key=resolved_settings.precog.api_key,
            timeout=resolved_settings.precog.timeout,
        )
        if resolved_settings.precog.enabled
        else None
    )
    memory_store = _build_memory_store(resolved_settings)
    app = create_app(
        registry=registry,
        router=router,
        proxy=proxy,
        collector=collector,
        feedback_loop=feedback_loop,
        evaluator_interval_seconds=resolved_settings.evaluator.collection.flush_interval_seconds
        if feedback_loop is not None
        else None,
        api_key=resolved_settings.server.api_key,
        cors_origins=resolved_settings.server.cors_origins,
        precog_publisher=precog_publisher,
        precog_project=resolved_settings.precog.project,
        memory_store=memory_store,
        health_tracker=health_tracker,
    )
    app_holder["app"] = app
    return app


def _build_memory_store(settings: Settings) -> SQLiteMemoryStore | PrecogMemoryStore | None:
    if not settings.memory.enabled:
        return None
    backend = settings.memory.backend.lower()
    if backend == "precog":
        return PrecogMemoryStore(
            PrecogMemoryConfig(
                enabled=True,
                base_url=settings.precog.base_url,
                api_key=settings.precog.api_key,
                timeout=settings.precog.timeout,
                default_project=settings.memory.default_project,
                top_k=settings.memory.top_k,
                min_score=settings.memory.min_score,
                max_context_chars=settings.memory.max_context_chars,
                query_path=settings.memory.query_path,
                record_path=settings.memory.record_path,
            )
        )
    return SQLiteMemoryStore(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=settings.memory.db_path,
            default_project=settings.memory.default_project,
            top_k=settings.memory.top_k,
            min_score=settings.memory.min_score,
            max_context_chars=settings.memory.max_context_chars,
            min_prompt_chars=settings.memory.min_prompt_chars,
            min_response_chars=settings.memory.min_response_chars,
        )
    )


def _ensure_runtime_logging(*, debug: bool) -> None:
    """Ensure LLMrouter logs are visible when loaded directly by Uvicorn."""
    logger = logging.getLogger("llmrouter")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)


def build_registry(models_file: str) -> ModelRegistry:
    """Load the model registry, returning an empty registry when no file exists."""
    path = _ensure_models_file(Path(models_file))
    if not path.exists():
        return ModelRegistry()
    return load_model_registry(path)


def _ensure_models_file(path: Path) -> Path:
    if path.exists():
        return path
    template = path.with_name(f"{path.stem}.example{path.suffix}")
    if not template.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, path)
    return path


def _priority_demoter(
    models_file: str,
    router: MultiModelRouter,
    provider_cooldowns: ProviderCooldownStore | None = None,
    app_holder: dict[str, FastAPI] | None = None,
    proxy_holder: dict[str, ProviderProxy] | None = None,
) -> Callable[[ModelInfo, ProviderError], None]:
    def demoter(model: ModelInfo, exc: ProviderError) -> None:
        if not _is_insufficient_balance_error(exc):
            return
        cooldown_entry = (
            provider_cooldowns.record_quota_error(model, exc)
            if provider_cooldowns is not None
            else None
        )
        if cooldown_entry is None:
            router.mark_provider_unavailable(model.provider)
            proxy = proxy_holder.get("proxy") if proxy_holder is not None else None
            if proxy is not None:
                proxy.disable_provider(model.provider)
            demoted = demote_model_priority(models_file, model.name)
            if demoted:
                registry = build_registry(models_file)
                router.replace_registry(registry)
                app = app_holder.get("app") if app_holder is not None else None
                if app is not None:
                    app.state.registry = registry
        logging.getLogger("llmrouter.runtime").warning(
            "%s provider '%s' for model '%s' after balance/quota error: %s",
            (
                "Put in quota cooldown"
                if cooldown_entry is not None
                else "Disabled"
            ),
            model.provider.value,
            model.name,
            exc,
        )

    return demoter


def _is_insufficient_balance_error(exc: ProviderError) -> bool:
    return is_quota_exhaustion_error(exc)


def build_providers(settings: Settings, registry: ModelRegistry) -> dict[Provider, BaseProvider]:
    """Instantiate providers required by the configured model catalog."""
    needed = {model.provider for model in registry.all()}
    providers: dict[Provider, BaseProvider] = {}

    if Provider.OLLAMA in needed and settings.providers.ollama.enabled:
        providers[Provider.OLLAMA] = _ollama(settings.providers.ollama)
    if Provider.OPENAI in needed and settings.providers.openai.enabled:
        api_key = resolve_api_key(settings.providers.openai, "OPENAI_API_KEY")
        if api_key:
            providers[Provider.OPENAI] = OpenAIProvider(
                api_key=api_key,
                base_url=settings.providers.openai.base_url,
                timeout=settings.providers.openai.timeout,
                max_retries=settings.providers.openai.max_retries,
            )
    if Provider.ZAI in needed and settings.providers.zai.enabled:
        api_key = resolve_api_key(settings.providers.zai, "ZAI_API_KEY")
        if api_key:
            providers[Provider.ZAI] = ZaiProvider(
                api_key=api_key,
                base_url=settings.providers.zai.base_url,
                timeout=settings.providers.zai.timeout,
                max_retries=settings.providers.zai.max_retries,
            )
    if Provider.GEMINI in needed and settings.providers.gemini.enabled:
        api_key = resolve_api_key(settings.providers.gemini, "GEMINI_API_KEY", "GOOGLE_API_KEY")
        if api_key:
            providers[Provider.GEMINI] = GeminiProvider(
                api_key=api_key,
                base_url=settings.providers.gemini.base_url,
                timeout=settings.providers.gemini.timeout,
                max_retries=settings.providers.gemini.max_retries,
            )
    if Provider.DEEPSEEK in needed and settings.providers.deepseek.enabled:
        api_key = resolve_api_key(settings.providers.deepseek, "DEEPSEEK_API_KEY")
        if api_key:
            providers[Provider.DEEPSEEK] = DeepSeekProvider(
                api_key=api_key,
                base_url=settings.providers.deepseek.base_url,
                timeout=settings.providers.deepseek.timeout,
                max_retries=settings.providers.deepseek.max_retries,
            )

    return providers


def _ollama(config: ProviderConfig) -> OllamaProvider:
    return OllamaProvider(
        api_key=resolve_api_key(config, "OLLAMA_API_KEY"),
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _build_health_tracker(settings: Settings) -> ModelHealthTracker:
    """Create a health tracker from settings, choosing the right backend."""
    weights = HealthWeights(
        latency=settings.health.latency_weight,
        error=settings.health.error_weight,
        quality=settings.health.quality_weight,
        cost=settings.health.cost_weight,
    )
    backend = HealthBackend(settings.health.backend.lower())
    if backend == HealthBackend.SQLITE:
        store = SQLiteHealthStore(
            db_path=settings.health.db_path,
            ttl_minutes=settings.health.ttl_minutes,
        )
    elif backend == HealthBackend.REDIS:
        # Redis backend not yet implemented; fall back to memory with a warning.
        logging.getLogger("llmrouter.runtime").warning(
            "Redis health backend is not implemented yet; falling back to in-memory store"
        )
        store = InMemoryHealthStore()
    else:
        store = InMemoryHealthStore()

    quality_source = None
    if settings.evaluator.enabled:
        quality_source = ReviewQualitySource(
            db_path=settings.evaluator.db_path,
            window_minutes=settings.health.ttl_minutes,
        )
    return ModelHealthTracker(
        store=store,
        window_minutes=settings.health.window_minutes,
        weights=weights,
        quality_source=quality_source,
    )


def _scorer_weights(raw_weights: dict[str, float]) -> ScorerWeights:
    return ScorerWeights(
        length=raw_weights.get("length", 0.15),
        code_detection=raw_weights.get("code_detection", 0.25),
        complexity_keywords=raw_weights.get("complexity_keywords", 0.20),
        math_detection=raw_weights.get("math_detection", 0.20),
        language_complexity=raw_weights.get("language_complexity", 0.20),
    )
