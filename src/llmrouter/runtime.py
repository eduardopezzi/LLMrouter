"""Runtime assembly from settings and model catalog."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from llmrouter.api.routes import create_app
from llmrouter.config import ProviderConfig, Settings, get_settings
from llmrouter.core.proxy import ProviderProxy
from llmrouter.core.registry import ModelRegistry, load_model_registry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer, ScorerWeights
from llmrouter.core.types import Provider
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


def build_app(settings: Settings | None = None) -> FastAPI:
    """Create a fully wired FastAPI app from configuration."""
    resolved_settings = settings or get_settings()
    _ensure_runtime_logging(debug=resolved_settings.debug)
    registry = build_registry(resolved_settings.models_file)
    proxy = ProviderProxy(build_providers(resolved_settings, registry))
    router = MultiModelRouter(
        registry=registry,
        scorer=PromptScorer(_scorer_weights(resolved_settings.routing.scorer_weights)),
        strategy=resolved_settings.routing.strategy,
        fallback_count=resolved_settings.routing.fallback_count,
        provider_cost_order=resolved_settings.routing.provider_cost_order,
    )
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
    return create_app(
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
    path = Path(models_file)
    if not path.exists():
        return ModelRegistry()
    return load_model_registry(path)


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
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _api_key(config: ProviderConfig, *env_names: str) -> str | None:
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
