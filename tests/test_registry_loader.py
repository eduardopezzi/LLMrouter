from __future__ import annotations

import shutil

import pytest

from llmrouter.core.registry import load_model_registry
from llmrouter.core.types import Provider, Tier
from llmrouter.runtime import build_registry


def test_load_model_registry_from_catalog() -> None:
    registry = load_model_registry("config/models.example.yaml")

    first = registry.get("ollama/glm-5.3-flash:cloud")
    assert first is not None
    assert first.provider == Provider.OLLAMA
    assert first.tier == Tier.T3
    assert "architecture" in first.capabilities
    assert first.cost_per_1k_input == pytest.approx(0.15 / 1000)
    assert first.api_base == "http://localhost:11434"
    assert first.max_tokens == 131072
    assert first.context_window == 1000000

    zhipu_flash = registry.get("zhipu/glm-5.3-flash")
    assert zhipu_flash is not None
    assert zhipu_flash.model_family == "glm-5.3-flash"
    assert zhipu_flash.tier == Tier.T1
    assert zhipu_flash in registry.by_tier(Tier.T1)
    assert zhipu_flash in registry.by_tier(Tier.T2)
    assert zhipu_flash not in registry.by_tier(Tier.T3)
    assert first in registry.by_tier(Tier.T3)
    assert first not in registry.by_tier(Tier.T1)


def test_load_model_registry_can_include_disabled_entries() -> None:
    active = load_model_registry("config/models.example.yaml")
    complete = load_model_registry("config/models.example.yaml", include_disabled=True)

    assert len(complete.models) > len(active.models)
    assert all(model.enabled for model in active.models)
    assert any(not model.enabled for model in complete.models)


def test_provider_model_name_removes_catalog_namespace() -> None:
    registry = build_registry("config/models.example.yaml")

    assert registry.get("ollama/qwen2.5-coder:3b").provider_model_name == "qwen2.5-coder:3b"
    assert (
        registry.get("deepseek/deepseek-flash").provider_model_name
        == "deepseek-flash"
    )
    assert registry.get("zhipu/glm-5.3").provider_model_name == "glm-5.3"


def test_build_registry_creates_local_models_file_from_example(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copyfile("config/models.example.yaml", config_dir / "models.example.yaml")

    registry = build_registry(str(config_dir / "models.yaml"))

    assert (config_dir / "models.yaml").exists()
    assert registry == load_model_registry(config_dir / "models.example.yaml")


@pytest.mark.parametrize("value", ["1", "[]", "[0]", "[4]", "[true]", "null", "[1, '2']"])
def test_invalid_routing_tiers_rejected(tmp_path, value) -> None:
    catalog = tmp_path / "models.yaml"
    catalog.write_text(f"models:\n  - name: flash\n    provider: zai\n    routing_tiers: {value}\n")
    with pytest.raises(ValueError, match="flash: routing_tiers"):
        load_model_registry(catalog)
