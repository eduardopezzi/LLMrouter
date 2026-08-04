from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llmrouter.benchmark_catalog import load_catalog_scores, load_catalog_source_urls
from llmrouter.core.types import ModelInfo, Provider, Tier


@dataclass(frozen=True)
class ModelRegistry:
    """Registry of available models for routing."""

    models: tuple[ModelInfo, ...] = ()

    def __contains__(self, model_name: str) -> bool:
        return any(model.name == model_name for model in self.models)

    def get(self, model_name: str) -> ModelInfo | None:
        for model in self.models:
            if model.name == model_name:
                return model
        return None

    def all(self) -> list[ModelInfo]:
        return list(self.models)

    def by_tier(self, tier: Tier | int) -> list[ModelInfo]:
        return [model for model in self.models if model.tier == tier]

    def add(self, model: ModelInfo) -> ModelRegistry:
        if model in self.models:
            return self
        return ModelRegistry(models=tuple(self.models) + (model,))

    def extend(self, models: Iterable[ModelInfo]) -> ModelRegistry:
        result = list(self.models)
        for model in models:
            if model not in result:
                result.append(model)
        return ModelRegistry(models=tuple(result))


def load_model_registry(
    path: str | Path,
    *,
    benchmark_catalog_path: str | Path | None = None,
) -> ModelRegistry:
    """Load model definitions from a YAML catalog."""
    data = _load_yaml(Path(path))
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("models file must contain a top-level 'models' list")
    refreshed_scores = load_catalog_scores(benchmark_catalog_path) if benchmark_catalog_path else {}
    refreshed_sources = (
        load_catalog_source_urls(benchmark_catalog_path) if benchmark_catalog_path else {}
    )
    models = [_model_from_mapping(item, refreshed_scores, refreshed_sources) for item in raw_models]
    return ModelRegistry(models=tuple(sorted(models, key=lambda model: model.priority)))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("models file must be a YAML mapping")
    return data


def _model_from_mapping(
    item: object,
    refreshed_scores: dict[str, dict[str, float]] | None = None,
    refreshed_sources: dict[str, tuple[str, ...]] | None = None,
) -> ModelInfo:
    if not isinstance(item, dict):
        raise ValueError("each model entry must be a mapping")

    name = _required_str(item, "name")
    provider = Provider(_required_str(item, "provider"))
    roles = _string_set(item.get("roles", []))
    max_tokens = int(item.get("max_tokens", item.get("context_window", 8192)))
    context_window = int(item.get("context_window", max_tokens))
    priority = int(item.get("priority", 10))
    tier = _parse_tier(item.get("tier"), roles, max_tokens, priority, name)
    # Let ModelInfo.__post_init__ validate the range; do NOT clamp silently.
    rollout_percentage = float(item.get("rollout_percentage", 100.0))

    configured_scores = dict(_benchmark_scores(item.get("benchmark_scores", {})))
    configured_scores.update((refreshed_scores or {}).get(name, {}))
    benchmark_sources = set(_url_list(item.get("benchmark_sources", [])))
    benchmark_sources.update((refreshed_sources or {}).get(name, ()))
    return ModelInfo(
        name=name,
        provider=provider,
        tier=tier,
        cost_per_1k_input=_cost_per_1k(item, "prompt_cost_per_1m_tokens"),
        cost_per_1k_output=_cost_per_1k(item, "completion_cost_per_1m_tokens"),
        max_tokens=max_tokens,
        capabilities=roles,
        priority=priority,
        context_window=context_window,
        api_base=_optional_str(item.get("api_base")),
        description=_optional_str(item.get("description")) or "",
        rollout_percentage=rollout_percentage,
        benchmark_scores=tuple(sorted(configured_scores.items())),
        benchmark_sources=tuple(sorted(benchmark_sources)),
    )


def _parse_tier(
    raw_tier: object,
    roles: frozenset[str],
    max_tokens: int,
    priority: int,
    name: str,
) -> Tier:
    if raw_tier is not None:
        return Tier(int(raw_tier))

    high_complexity_roles = {"architecture", "security_audit", "review", "migration"}
    simple_roles = {"summarization", "documentation"}
    lowered_name = name.lower()

    if "3b" in lowered_name or "nano" in lowered_name:
        return Tier.T1
    if roles and roles <= simple_roles and max_tokens <= 32768:
        return Tier.T1
    if roles & high_complexity_roles or max_tokens >= 128000 or priority <= 4:
        return Tier.T3
    return Tier.T2


def _cost_per_1k(item: dict[str, object], key: str) -> float:
    raw = item.get(key, 0)
    return float(raw or 0) / 1000


def _required_str(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model entry missing required string field: {key}")
    return value


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_set(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(str(item) for item in value if isinstance(item, str))


def _url_list(value: object) -> tuple[str, ...]:
    """Parse optional, review-approved HTTPS benchmark links from model YAML."""
    if not isinstance(value, list):
        return ()
    return tuple(sorted({item for item in value if isinstance(item, str) and item.startswith("https://")}))


def _benchmark_scores(value: object) -> tuple[tuple[str, float], ...]:
    """Parse optional raw benchmark measurements from a model catalog entry."""
    if not isinstance(value, dict):
        return ()
    parsed: list[tuple[str, float]] = []
    for name, score in value.items():
        if (
            not isinstance(name, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float, str))
        ):
            continue
        try:
            parsed.append((name, float(score)))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(parsed))
