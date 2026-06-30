from __future__ import annotations

from llmrouter.core.registry import load_model_registry
from llmrouter.core.types import Provider, Tier
from llmrouter.runtime import build_registry


def test_load_model_registry_from_catalog() -> None:
    registry = load_model_registry("config/models.yaml")

    assert len(registry.models) == 22
    first = registry.models[0]
    assert first.name == "zhipu/glm-5.2"
    assert first.provider == Provider.ZAI
    assert first.tier == Tier.T3
    assert "review" in first.capabilities
    assert first.cost_per_1k_input == 0
    assert first.api_base is None


def test_provider_model_name_removes_catalog_namespace() -> None:
    registry = build_registry("config/models.yaml")

    assert registry.get("ollama/qwen2.5-coder:3b").provider_model_name == "qwen2.5-coder:3b"
    assert (
        registry.get("nvidia_nim/moonshotai/kimi-k2.6").provider_model_name
        == "moonshotai/kimi-k2.6"
    )
    assert registry.get("zhipu/glm-5.2").provider_model_name == "glm-5.2"
