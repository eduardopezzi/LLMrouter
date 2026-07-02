"""Runtime assembly from settings and model catalog."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI

from llmrouter.api.routes import create_app
from llmrouter.cli_panel import demote_model_priority
from llmrouter.config import ProviderConfig, Settings, get_settings
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry, load_model_registry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer, ScorerWeights
from llmrouter.core.types import ModelInfo, Provider
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.feedback import FeedbackLoop
from llmrouter.evaluator.grader import RoutingDecisionGrader
from llmrouter.evaluator.judge import QualityJudge
from llmrouter.precog import PrecogPublisher
from llmrouter.providers import (
    BaseProvider,
    GeminiProvider,
    NvidiaProvider,
    OllamaProvider,
    OpenAIProvider,
    ZaiProvider,
)
from llmrouter.providers.base import ProviderError


class _ApiKeyConfig(Protocol):
    api_key: str | None


def build_app(settings: Settings | None = None) -> FastAPI:
    """Create a fully wired FastAPI app from configuration."""
    resolved_settings = settings or get_settings()
    _ensure_runtime_logging(debug=resolved_settings.debug)
    registry = build_registry(resolved_settings.models_file)
    router = MultiModelRouter(
        registry=registry,
        scorer=PromptScorer(_scorer_weights(resolved_settings.routing.scorer_weights)),
        strategy=resolved_settings.routing.strategy,
        fallback_count=resolved_settings.routing.fallback_count,
        provider_cost_order=resolved_settings.routing.provider_cost_order,
    )
    app_holder: dict[str, FastAPI] = {}
    proxy_holder: dict[str, ProviderProxy] = {}
    proxy = ProviderProxy(
        build_providers(resolved_settings, registry),
        on_provider_error=_priority_demoter(
            resolved_settings.models_file,
            router,
            app_holder,
            proxy_holder,
        ),
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
                api_key=_api_key(resolved_settings.evaluator.ollama, "OLLAMA_API_KEY"),
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
    )
    app_holder["app"] = app
    return app


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
    app_holder: dict[str, FastAPI] | None = None,
    proxy_holder: dict[str, ProviderProxy] | None = None,
) -> Callable[[ModelInfo, ProviderError], None]:
    def demoter(model: ModelInfo, exc: ProviderError) -> None:
        if not _is_insufficient_balance_error(exc):
            return
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
            "Disabled provider '%s' and %s model '%s' after balance/quota error: %s",
            model.provider.value,
            "demoted" if demoted else "kept lowest-priority",
            model.name,
            exc,
        )

    return demoter


def _is_insufficient_balance_error(exc: ProviderError) -> bool:
    if exc.status_code not in {402, 429}:
        return False
    message = str(exc).lower()
    indicators = (
        "余额不足",
        "无可用资源包",
        "请充值",
        "insufficient balance",
        "insufficient quota",
        "insufficient credits",
        "no available resource",
        "quota exceeded",
        "billing",
        "credit",
        "credits",
        "recharge",
    )
    return any(indicator in message for indicator in indicators)


def build_providers(settings: Settings, registry: ModelRegistry) -> dict[Provider, BaseProvider]:
    """Instantiate providers required by the configured model catalog."""
    needed = {model.provider for model in registry.all()}
    providers: dict[Provider, BaseProvider] = {}

    if Provider.OLLAMA in needed and settings.providers.ollama.enabled:
        providers[Provider.OLLAMA] = _ollama(settings.providers.ollama)
    if Provider.OPENAI in needed and settings.providers.openai.enabled:
        api_key = _api_key(settings.providers.openai, "OPENAI_API_KEY")
        if api_key:
            providers[Provider.OPENAI] = OpenAIProvider(
                api_key=api_key,
                base_url=settings.providers.openai.base_url,
                timeout=settings.providers.openai.timeout,
                max_retries=settings.providers.openai.max_retries,
            )
    if Provider.NVIDIA in needed and settings.providers.nvidia.enabled:
        api_key = _api_key(settings.providers.nvidia, "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY")
        if api_key:
            providers[Provider.NVIDIA] = NvidiaProvider(
                api_key=api_key,
                base_url=settings.providers.nvidia.base_url,
                timeout=settings.providers.nvidia.timeout,
                max_retries=settings.providers.nvidia.max_retries,
            )
    if Provider.ZAI in needed and settings.providers.zai.enabled:
        api_key = _api_key(settings.providers.zai, "ZAI_API_KEY")
        if api_key:
            providers[Provider.ZAI] = ZaiProvider(
                api_key=api_key,
                base_url=settings.providers.zai.base_url,
                timeout=settings.providers.zai.timeout,
                max_retries=settings.providers.zai.max_retries,
            )
    if Provider.GEMINI in needed and settings.providers.gemini.enabled:
        api_key = _api_key(settings.providers.gemini, "GEMINI_API_KEY", "GOOGLE_API_KEY")
        if api_key:
            providers[Provider.GEMINI] = GeminiProvider(
                api_key=api_key,
                base_url=settings.providers.gemini.base_url,
                timeout=settings.providers.gemini.timeout,
                max_retries=settings.providers.gemini.max_retries,
            )

    return providers


def _ollama(config: ProviderConfig) -> OllamaProvider:
    return OllamaProvider(
        api_key=_api_key(config, "OLLAMA_API_KEY"),
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _api_key(config: _ApiKeyConfig, *env_names: str) -> str | None:
    if config.api_key:
        return config.api_key
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _scorer_weights(raw_weights: dict[str, float]) -> ScorerWeights:
    return ScorerWeights(
        length=raw_weights.get("length", 0.15),
        code_detection=raw_weights.get("code_detection", 0.25),
        complexity_keywords=raw_weights.get("complexity_keywords", 0.20),
        math_detection=raw_weights.get("math_detection", 0.20),
        language_complexity=raw_weights.get("language_complexity", 0.20),
    )
