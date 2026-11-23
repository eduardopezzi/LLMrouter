"""Tests for review-only local model catalog discovery."""

from __future__ import annotations

import json

from llmrouter.core.registry import load_model_registry
from llmrouter.core.types import ModelInfo, Provider, Tier
from llmrouter.model_catalog import reconcile_ollama_local_models, write_catalog_proposals


def _model(name: str) -> ModelInfo:
    return ModelInfo(name=name, provider=Provider.OLLAMA, tier=Tier.T2)


def test_inkling_cloud_is_catalogued_with_vendor_benchmarks_and_safe_rollout() -> None:
    registry = load_model_registry(
        "config/models.example.yaml",
        benchmark_catalog_path="data/model_benchmarks.yaml",
    )
    model = registry.get("ollama/inkling-2-small:cloud")

    assert model is not None
    assert model.provider is Provider.OLLAMA
    assert model.tier is Tier.T3
    assert model.rollout_percentage == 0
    assert model.context_window == 1_000_000
    assert model.cost_per_1k_input == 0.0003
    assert model.cost_per_1k_output == 0.001
    assert {"review", "fix", "test_generation"} <= model.capabilities
    assert dict(model.benchmark_scores)["DeepSWE v1.1"] == 72.1
    assert "https://thinkingmachines.ai/news/inkling-2-small/" in model.benchmark_sources


def test_reconcile_proposes_add_and_local_absence_but_ignores_cloud() -> None:
    proposals = reconcile_ollama_local_models(
        [
            _model("ollama/qwen3:8b"),
            _model("ollama/kimi-k2.7-code:cloud"),
        ],
        {"qwen3:14b"},
    )

    assert [(proposal.action, proposal.model) for proposal in proposals] == [
        ("add", "ollama/qwen3:14b"),
        ("verify_absent", "ollama/qwen3:8b"),
    ]


def test_proposals_are_written_without_touching_model_yaml(tmp_path) -> None:
    output = tmp_path / "catalog_proposals.json"
    proposal = reconcile_ollama_local_models([], {"gemma3:4b"})

    report = write_catalog_proposals(
        output,
        configured_models=0,
        inventory={"gemma3:4b"},
        proposals=proposal,
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report.output_path == output
    assert saved["status"] == "pending_human_review"
    assert saved["proposals"][0]["model"] == "ollama/gemma3:4b"
    assert saved["proposals"][0]["suggested_rollout_percentage"] == 0.0
