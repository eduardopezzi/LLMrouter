from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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

    def by_tier(self, tier: int) -> list[ModelInfo]:
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


def load_model_registry(path: str | Path) -> ModelRegistry:
    """Load model definitions from a YAML catalog."""
    data = _load_yaml(Path(path))
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("models file must contain a top-level 'models' list")
    models = [_model_from_mapping(item) for item in raw_models]
    return ModelRegistry(models=tuple(sorted(models, key=lambda model: model.priority)))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("models file must be a YAML mapping")
    return data


def _model_from_mapping(item: object) -> ModelInfo:
    if not isinstance(item, dict):
        raise ValueError("each model entry must be a mapping")

    name = _required_str(item, "name")
    provider = Provider(_required_str(item, "provider"))
    roles = _string_set(item.get("roles", []))
    max_tokens = int(item.get("max_tokens", item.get("context_window", 8192)))
    context_window = int(item.get("context_window", max_tokens))
    priority = int(item.get("priority", 10))
    tier = _parse_tier(item.get("tier"), roles, max_tokens, priority, name)

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
