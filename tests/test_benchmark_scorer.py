from __future__ import annotations

import math

import pytest

from llmrouter.cli_panel import _extract_json_object
from llmrouter.cli_panel import _extract_json_object as extract_json
from llmrouter.core.benchmark_scorer import (
    BENCHMARK_WEIGHTS,
    _benchmark_quality_score,
    _context_window_score,
    _cost_score,
    _lookup_benchmark_scores,
    _provider_multiplier,
    _tier_score,
    rank_models,
    score_model,
)
from llmrouter.core.types import ModelInfo, Provider, Tier


def test_benchmark_weights_sum_to_one() -> None:
    assert math.isclose(sum(BENCHMARK_WEIGHTS.values()), 1.0)


def test_lookup_benchmark_scores_matches_name_fragment() -> None:
    scores = _lookup_benchmark_scores("ollama/deepseek-v4-pro:cloud")
    assert scores is not None
    assert "mmlu" in scores
    assert 0 < scores["mmlu"] <= 1


def test_lookup_benchmark_scores_returns_none_for_unknown_model() -> None:
    assert _lookup_benchmark_scores("unknown/foobar") is None


def test_benchmark_quality_score_ignores_missing_benchmarks() -> None:
    scores = {"mmlu": 0.8, "gpqa": 0.6}
    result = _benchmark_quality_score(scores)
    expected = (0.8 * 0.35 + 0.6 * 0.15) / (0.35 + 0.15)
    assert math.isclose(result, expected, rel_tol=1e-4)


def test_benchmark_quality_score_returns_zero_when_empty() -> None:
    assert _benchmark_quality_score({}) == 0.0
    assert _benchmark_quality_score(None) == 0.0


def test_tier_score_normalizes_values() -> None:
    assert _tier_score(Tier.T1) == pytest.approx(1 / 3)
    assert _tier_score(Tier.T2) == pytest.approx(2 / 3)
    assert _tier_score(Tier.T3) == 1.0


def test_context_window_score_caps_at_one() -> None:
    assert _context_window_score(0) == 0.0
    assert _context_window_score(1024) < _context_window_score(8192)
    assert _context_window_score(2_000_000) == 1.0


def test_cost_score_prefers_cheaper_models() -> None:
    assert _cost_score(0.0) == 1.0
    assert _cost_score(10.0) == 0.5
    assert _cost_score(90.0) < _cost_score(10.0)


def test_provider_multiplier_uses_cost_order() -> None:
    order = ["nvidia", "ollama", "zai"]
    assert _provider_multiplier(Provider.NVIDIA, order) == 1.0
    assert _provider_multiplier(Provider.OLLAMA, order) == 0.97
    assert _provider_multiplier(Provider.ZAI, order) == 0.94
    assert _provider_multiplier(Provider.GEMINI, order) == 0.70


def test_score_model_returns_breakdown() -> None:
    model = ModelInfo(
        name="ollama/deepseek-v4-pro:cloud",
        provider=Provider.OLLAMA,
        tier=Tier.T3,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        context_window=1_000_000,
        max_tokens=1_000_000,
    )
    score = score_model(model, strategy="balanced", provider_cost_order=["ollama"])

    assert score.name == model.name
    assert 0 < score.benchmark_score <= 1
    assert score.tier_score == 1.0
    assert score.context_window_score == 1.0
    assert 0 < score.cost_score <= 1
    assert score.provider_multiplier == 1.0
    assert 0 < score.strategy_score <= 1


def test_score_model_falls_back_to_tier_when_no_benchmark() -> None:
    model = ModelInfo(
        name="custom/unknown-model",
        provider=Provider.OLLAMA,
        tier=Tier.T2,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    )
    score = score_model(model, strategy="quality", provider_cost_order=["ollama"])
    assert score.benchmark_score == 0.0
    assert score.tier_score == pytest.approx(2 / 3, abs=1e-4)
    assert score.strategy_score > 0


def test_rank_models_orders_by_strategy_score_descending() -> None:
    models = [
        ModelInfo(
            name="ollama/qwen2.5-coder:3b",
            provider=Provider.OLLAMA,
            tier=Tier.T1,
            priority=1,
        ),
        ModelInfo(
            name="ollama/deepseek-v4-pro:cloud",
            provider=Provider.OLLAMA,
            tier=Tier.T3,
            priority=2,
        ),
        ModelInfo(
            name="ollama/kimi-k2.7-code:cloud",
            provider=Provider.OLLAMA,
            tier=Tier.T3,
            priority=3,
        ),
    ]
    ordered = rank_models(models, strategy="quality", provider_cost_order=["ollama"])
    assert ordered[0].name in {
        "ollama/deepseek-v4-pro:cloud",
        "ollama/kimi-k2.7-code:cloud",
    }
    assert ordered[-1].name == "ollama/qwen2.5-coder:3b"


def test_extract_json_object_fenced_and_plain() -> None:
    assert extract_json('{"a":1}') == '{"a":1}'
    assert extract_json("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert extract_json("x {\"a\":1} y") == '{"a":1}'
    assert extract_json("no json") is None
