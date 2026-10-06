"""Tests for review-only local model catalog discovery."""

from __future__ import annotations

import json

from llmrouter.core.types import ModelInfo, Provider, Tier
from llmrouter.model_catalog import reconcile_ollama_local_models, write_catalog_proposals


def _model(name: str) -> ModelInfo:
    return ModelInfo(name=name, provider=Provider.OLLAMA, tier=Tier.T2)


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
