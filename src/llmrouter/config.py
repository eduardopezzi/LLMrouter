"""Application configuration via pydantic-settings.

Supports loading from environment variables (.env) and YAML files.
All settings are type-safe and validated at startup.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from llmrouter.core.types import RoutingStrategy

# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------


class LogLevel(str, Enum):
    """Logging verbosity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = 12345
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key: str | None = Field(
        default=None,
        description="Gateway API key. If set, clients must send it via Authorization header.",
    )


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_retries: int = 3


class ProvidersConfig(BaseModel):
    """Aggregated provider configurations."""

    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(base_url="http://localhost:11434")
    )
    zai: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)


class RoutingConfig(BaseModel):
    """Routing engine configuration."""

    strategy: RoutingStrategy = RoutingStrategy.COST
    fallback_count: int = 2
    provider_cost_order: list[str] = Field(
        default_factory=lambda: ["zai", "ollama"],
        description="Provider preference used to break cost ties.",
    )
    max_cost_per_request: float | None = None
    scorer_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "length": 0.15,
            "code_detection": 0.25,
            "complexity_keywords": 0.20,
            "math_detection": 0.20,
            "language_complexity": 0.20,
        }
    )


class EvaluatorOllamaConfig(BaseModel):
    """Local Ollama model used for evaluation only."""

    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    model: str = "qwen2.5-coder:3b"
    timeout: float = 60.0
    temperature: float = 0.1


class EvaluatorCollectionConfig(BaseModel):
    """Observation collection settings."""

    buffer_size: int = 100
    flush_interval_seconds: int = 60
    sample_rate: float = 0.1


class EvaluatorFeedbackConfig(BaseModel):
    """Feedback loop settings."""

    min_samples_for_update: int = 50
    update_interval_hours: int = 24


class EvaluatorSyntheticConfig(BaseModel):
    """Synthetic data generation settings."""

    prompts_per_cycle: int = 20


class EvaluatorConfig(BaseModel):
    """Self-evaluation module configuration."""

    enabled: bool = True
    db_path: str = "data/llmrouter.db"
    ollama: EvaluatorOllamaConfig = Field(default_factory=EvaluatorOllamaConfig)
    collection: EvaluatorCollectionConfig = Field(default_factory=EvaluatorCollectionConfig)
    feedback: EvaluatorFeedbackConfig = Field(default_factory=EvaluatorFeedbackConfig)
    synthetic: EvaluatorSyntheticConfig = Field(default_factory=EvaluatorSyntheticConfig)


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = True
    endpoint: str = "/metrics"


class PrecogConfig(BaseModel):
    """PRecog integration for routing observations and feedback."""

    enabled: bool = False
    base_url: str = "http://localhost:8888"
    api_key: str | None = None
    project: str = "llmrouter"
    timeout: float = 3.0


class MemoryConfig(BaseModel):
    """Local project memory/RAG settings."""

    enabled: bool = False
    backend: str = "local"  # local | precog
    db_path: str = "data/llmrouter_memory.db"
    default_project: str = "default"
    top_k: int = 4
    min_score: float = 0.12
    max_context_chars: int = 2400
    min_prompt_chars: int = 80
    min_response_chars: int = 40
    query_path: str = "/internal/rag/query"
    record_path: str = "/internal/llmrouter/observations"


class HealthConfig(BaseModel):
    """Model health tracking configuration."""

    enabled: bool = True
    backend: str = "memory"  # memory | sqlite | redis
    db_path: str = "data/health.db"
    window_minutes: int = 15
    ttl_minutes: int = 60
    latency_weight: float = 0.30
    error_weight: float = 0.35
    quality_weight: float = 0.25
    cost_weight: float = 0.10


class SemanticConfig(BaseModel):
    """Semantic prompt scoring configuration."""

    enabled: bool = False
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # cpu | cuda | mps
    cache_dir: str | None = None
    embedding_cache_path: str = "data/semantic_role_embeddings.json"
    fallback_to_rule_based: bool = True


class HybridScorerConfig(BaseModel):
    """Hybrid scorer weights between rule-based and semantic scorers."""

    rule_weight: float = 0.30
    semantic_weight: float = 0.70
    semantic_confidence_threshold: float = 0.35


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root application settings.

    Values are loaded from (in priority order):
    1. Environment variables (prefix ``LLMROUTER_``)
    2. ``.env`` file
    3. YAML config file (``config/config.yaml`` or path in ``LLMROUTER_CONFIG``)
    4. Built-in defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="LLMROUTER_",
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    app_name: str = "LLMrouter"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    # Sub-configs
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    precog: PrecogConfig = Field(default_factory=PrecogConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    hybrid: HybridScorerConfig = Field(default_factory=HybridScorerConfig)

    # Model registry file
    models_file: str = "config/models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if it doesn't exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings.

    Loads from YAML first, then applies environment overrides.
    """
    import os

    config_path = os.environ.get("LLMROUTER_CONFIG", "config/config.yaml")
    yaml_data = _load_yaml(Path(config_path))

    # Build Settings from YAML base, then env overrides
    settings = Settings(**yaml_data) if yaml_data else Settings()
    return settings


def reload_settings() -> Settings:
    """Force reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()