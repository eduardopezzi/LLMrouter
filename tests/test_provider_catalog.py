"""Tests for the weekly provider documentation/model monitor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import llmrouter.provider_catalog as provider_catalog
from llmrouter.core.registry import load_model_registry
from llmrouter.core.types import ModelInfo, Provider, Tier


def test_parse_model_api_supports_ollama_and_openai_shapes() -> None:
    assert provider_catalog._parse_model_api(
        "ollama", {"models": [{"name": "glm-5.3-flash:cloud"}]}
    ) == [{"id": "glm-5.3-flash:cloud"}]
    assert provider_catalog._parse_model_api(
        "deepseek", {"data": [{"id": "deepseek-v4-flash", "owned_by": "deepseek"}]}
    ) == [{"id": "deepseek-v4-flash", "owned_by": "deepseek"}]


@pytest.mark.parametrize("payload", [{}, {"error": "unauthorized"}])
def test_parse_model_api_rejects_success_responses_without_inventory(payload) -> None:
    with pytest.raises(ValueError, match="no model list"):
        provider_catalog._parse_model_api("deepseek", payload)


def test_documentation_models_are_conservative_and_normalized() -> None:
    found = provider_catalog._extract_documentation_models(
        "deepseek", "deepseek-v4-flash-vision-exp and not-a-model"
    )
    assert found == [{"id": "deepseek-v4-flash-vision-exp"}]
    assert provider_catalog._extract_documentation_models("ollama", "glm-5.3") == []


def test_model_diffs_do_not_repeat_proposals_or_infer_removals_on_first_run() -> None:
    models = {
        "deepseek": [
            ModelInfo("deepseek/deepseek-v4-flash", Provider.DEEPSEEK, Tier.T3),
        ]
    }
    discovered = {
        "deepseek": {
            "deepseek-v4-flash-vision-exp": {"id": "deepseek-v4-flash-vision-exp"}
        }
    }
    source_records = {
        "https://api.deepseek.com/models": {
            "provider": "deepseek",
            "kind": "model_api",
            "models": [{"id": "deepseek-v4-flash-vision-exp"}],
        }
    }
    new, removed, _ = provider_catalog._model_diffs(models, discovered, {}, source_records)
    assert [item["model"] for item in new] == ["deepseek/deepseek-v4-flash-vision-exp"]
    assert removed == []


    previous = {
        **source_records,
        "https://docs.example/models": {
            "provider": "deepseek",
            "kind": "documentation",
            "models": [{"id": "deepseek-v4-flash-vision-exp"}],
        },
    }
    new, removed, _ = provider_catalog._model_diffs(models, discovered, previous, source_records)
    assert new == []
    assert removed == []


def test_new_models_are_inserted_before_top_level_defaults(tmp_path: Path) -> None:
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        "models:\n"
        "  - name: zhipu/model-a\n"
        "    provider: zai\n"
        "    enabled: true\n"
        "    priority: 1\n"
        "defaults:\n"
        "  temperature: 0.1\n"
        "  max_retries: 2\n",
        encoding="utf-8",
    )

    added, reactivated = provider_catalog._apply_provider_catalog_changes(
        models_path,
        [],
        {"zai": {"glm-new": {"id": "glm-new"}}},
    )

    assert [item["model"] for item in added] == ["zhipu/glm-new"]
    assert reactivated == []
    data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in data["models"]] == [
        "zhipu/model-a",
        "zhipu/glm-new",
    ]
    assert data["defaults"] == {"temperature": 0.1, "max_retries": 2}
    assert load_model_registry(models_path).get("zhipu/glm-new") is not None


def test_refresh_writes_report_and_applies_only_priority_order(
    tmp_path: Path, monkeypatch
) -> None:
    sources_path = tmp_path / "sources.yaml"
    models_path = tmp_path / "models.yaml"
    snapshot_path = tmp_path / "snapshot.yaml"
    report_path = tmp_path / "report.json"
    sources_path.write_text(
        yaml.safe_dump(
            {"sources": [{"provider": "zai", "url": "https://docs.example/models"}]}
        ),
        encoding="utf-8",
    )
    models_path.write_text(
        """models:
  - name: zhipu/model-a
    provider: zai
    tier: 1
    priority: 1
  - name: zhipu/model-b
    provider: zai
    tier: 1
    priority: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        provider_catalog,
        "_fetch_source",
        lambda source, *, timeout: ("same body", [{"id": "glm-new"}]),
    )
    monkeypatch.setattr(
        provider_catalog,
        "rank_models",
        lambda models, **_: list(reversed(models)),
    )

    report = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_priority=True,
    )
    assert report.changed is True
    assert report.new_models[0]["model"] == "zhipu/glm-new"
    assert report.priority_changes
    assert report_path.exists()
    assert snapshot_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["priority_order"] == ["zhipu/model-b", "zhipu/model-a"]
    catalog = models_path.read_text(encoding="utf-8")
    assert "name: zhipu/model-b\n    provider: zai\n    tier: 1\n    priority: 1" in catalog
    assert "name: zhipu/model-a\n    provider: zai\n    tier: 1\n    priority: 2" in catalog


def test_refresh_applies_new_models_and_retires_after_two_inventory_checks(
    tmp_path: Path, monkeypatch
) -> None:
    sources_path = tmp_path / "sources.yaml"
    models_path = tmp_path / "models.yaml"
    snapshot_path = tmp_path / "snapshot.yaml"
    report_path = tmp_path / "report.json"
    sources_path.write_text(
        "sources:\n"
        "  - provider: zai\n"
        "    kind: model_api\n"
        "    url: https://provider.example/models\n"
        "  - provider: zai\n"
        "    url: https://provider.example/models-secondary\n"
        "    complete_inventory: true\n",
        encoding="utf-8",
    )
    models_path.write_text(
        "# Keep this catalog comment.\n"
        "models:\n"
        "  - name: zhipu/model-a\n"
        "    provider: zai\n"
        "    tier: 2\n"
        "    priority: 1\n"
        "  - name: zhipu/model-b\n"
        "    provider: zai\n"
        "    tier: 2\n"
        "    priority: 2\n",
        encoding="utf-8",
    )
    available_ids = ["model-a", "glm-new"]
    monkeypatch.setattr(
        provider_catalog,
        "_fetch_source",
        lambda source, *, timeout: ("stable inventory", [{"id": item} for item in available_ids]),
    )

    first = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_catalog=True,
    )
    assert [item["model"] for item in first.new_models] == ["zhipu/glm-new"]
    assert first.removed_models == ()
    discovered_model = load_model_registry(models_path).get("zhipu/glm-new")
    assert discovered_model is not None
    assert discovered_model.rollout_percentage == 0

    monkeypatch.setattr(
        provider_catalog,
        "_fetch_source",
        lambda source, *, timeout: ("empty inventory", []),
    )
    empty = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_catalog=True,
    )
    assert empty.source_errors
    assert empty.removed_models == ()
    assert "zhipu/model-b" in {model.name for model in load_model_registry(models_path).all()}

    def fail_source(source, *, timeout):
        if source.url.endswith("models-secondary"):
            raise provider_catalog.httpx.ConnectError("provider temporarily unavailable")
        return "stable inventory", [{"id": item} for item in available_ids]

    monkeypatch.setattr(provider_catalog, "_fetch_source", fail_source)
    failed = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_catalog=True,
    )
    assert failed.source_errors
    assert failed.removed_models == ()
    assert "zhipu/model-b" in {model.name for model in load_model_registry(models_path).all()}

    monkeypatch.setattr(
        provider_catalog,
        "_fetch_source",
        lambda source, *, timeout: ("stable inventory", [{"id": item} for item in available_ids]),
    )
    second = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_catalog=True,
    )
    assert second.removed_models == ()

    third = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_catalog=True,
    )
    assert [item["model"] for item in third.removed_models] == ["zhipu/model-b"]
    active_names = {model.name for model in load_model_registry(models_path).all()}
    assert active_names == {"zhipu/model-a", "zhipu/glm-new"}
    catalog = models_path.read_text(encoding="utf-8")
    assert "# Keep this catalog comment." in catalog
    assert "  - name: zhipu/model-b\n    enabled: false" in catalog

    available_ids[:] = ["model-a", "model-b", "glm-new"]
    fourth = provider_catalog.refresh_provider_catalog(
        sources_path,
        models_path,
        snapshot_path,
        report_path,
        strategy="balanced",
        provider_cost_order=["zai"],
        apply_catalog=True,
    )
    assert [item["model"] for item in fourth.reactivated_models] == ["zhipu/model-b"]
    assert {model.name for model in load_model_registry(models_path).all()} == {
        "zhipu/model-a",
        "zhipu/model-b",
        "zhipu/glm-new",
    }
