"""Benchmark-based scoring for model priority ranking.

This module provides a deterministic, reusable way to rank models using
public benchmark scores as quality signals, combined with catalog metadata
(tier, context window, cost) and the configured provider cost order.

Scores are normalized to the 0.0-1.0 range where possible. Missing benchmark
values fall back to the model's tier so that every model has a usable score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from llmrouter.core.types import ModelInfo, Provider, Tier


# Normalized benchmark scores (0.0-1.0) keyed by canonical model name fragment.
# These are illustrative defaults based on widely reported leaderboard results.
# Users can override/extend via the BENCHMARK_SCORES mapping below.
BENCHMARK_SCORES: dict[str, dict[str, float]] = {
    # Ollama models
    "kimi-k2.7-code": {
        "mmlu": 0.91,
        "humaneval": 0.89,
        "gpqa": 0.82,
        "mt_bench": 0.88,
        "ruler": 0.85,
    },
    "deepseek-v4-pro": {
        "mmlu": 0.93,
        "humaneval": 0.91,
        "gpqa": 0.88,
        "mt_bench": 0.90,
        "ruler": 0.87,
    },
    "deepseek-v4-flash": {
        "mmlu": 0.89,
        "humaneval": 0.86,
        "gpqa": 0.80,
        "mt_bench": 0.85,
        "ruler": 0.84,
    },
    "qwen3-coder": {
        "mmlu": 0.88,
        "humaneval": 0.87,
        "gpqa": 0.78,
        "mt_bench": 0.84,
        "ruler": 0.82,
    },
    "glm-5.2": {
        "mmlu": 0.85,
        "humaneval": 0.84,
        "gpqa": 0.74,
        "mt_bench": 0.83,
        "ruler": 0.80,
    },
    "north-mini-code": {
        "mmlu": 0.76,
        "humaneval": 0.78,
        "gpqa": 0.62,
        "mt_bench": 0.75,
        "ruler": 0.70,
    },
    "qwen3.6": {
        "mmlu": 0.83,
        "humaneval": 0.80,
        "gpqa": 0.70,
        "mt_bench": 0.80,
        "ruler": 0.78,
    },
    "gemma4:31b": {
        "mmlu": 0.86,
        "humaneval": 0.84,
        "gpqa": 0.75,
        "mt_bench": 0.83,
        "ruler": 0.81,
    },
    "qwen2.5-coder:3b": {
        "mmlu": 0.68,
        "humaneval": 0.72,
        "gpqa": 0.50,
        "mt_bench": 0.70,
        "ruler": 0.60,
    },
    "deepseek-v3.2": {
        "mmlu": 0.88,
        "humaneval": 0.87,
        "gpqa": 0.80,
        "mt_bench": 0.86,
        "ruler": 0.83,
    },
    "deepseek-v3.1": {
        "mmlu": 0.86,
        "humaneval": 0.84,
        "gpqa": 0.76,
        "mt_bench": 0.84,
        "ruler": 0.80,
    },
    # NVIDIA NIM models (use same upstream model names as lookup keys)
    "nemotron-3-ultra-550b": {
        "mmlu": 0.90,
        "humaneval": 0.87,
        "gpqa": 0.84,
        "mt_bench": 0.87,
        "ruler": 0.82,
    },
    "kimi-k2.6": {
        "mmlu": 0.90,
        "humaneval": 0.88,
        "gpqa": 0.83,
        "mt_bench": 0.88,
        "ruler": 0.84,
    },
    "deepseek-ai/deepseek-v4-pro": {
        "mmlu": 0.93,
        "humaneval": 0.91,
        "gpqa": 0.88,
        "mt_bench": 0.90,
        "ruler": 0.87,
    },
    "gemma-4-31b-it": {
        "mmlu": 0.86,
        "humaneval": 0.84,
        "gpqa": 0.75,
        "mt_bench": 0.83,
        "ruler": 0.81,
    },
    "deepseek-ai/deepseek-v4-flash": {
        "mmlu": 0.89,
        "humaneval": 0.86,
        "gpqa": 0.80,
        "mt_bench": 0.85,
        "ruler": 0.84,
    },
    "stepfun-ai/step-3.7-flash": {
        "mmlu": 0.82,
        "humaneval": 0.80,
        "gpqa": 0.70,
        "mt_bench": 0.80,
        "ruler": 0.78,
    },
    "mistralai/mistral-medium-3.5-128b": {
        "mmlu": 0.85,
        "humaneval": 0.83,
        "gpqa": 0.74,
        "mt_bench": 0.83,
        "ruler": 0.80,
    },
    "minimaxai/minimax-m3": {
        "mmlu": 0.84,
        "humaneval": 0.82,
        "gpqa": 0.72,
        "mt_bench": 0.82,
        "ruler": 0.78,
    },
    "nemotron-3-nano-omni-30b": {
        "mmlu": 0.75,
        "humaneval": 0.74,
        "gpqa": 0.60,
        "mt_bench": 0.74,
        "ruler": 0.68,
    },
    # Zhipu models
    "zhipu/glm-5.2": {
        "mmlu": 0.85,
        "humaneval": 0.84,
        "gpqa": 0.74,
        "mt_bench": 0.83,
        "ruler": 0.80,
    },
    "zhipu/glm-5.1": {
        "mmlu": 0.82,
        "humaneval": 0.80,
        "gpqa": 0.70,
        "mt_bench": 0.80,
        "ruler": 0.76,
    },
}

# Which benchmarks participate in the composite quality score and with what weight.
BENCHMARK_WEIGHTS: dict[str, float] = {
    "mmlu": 0.35,
    "humaneval": 0.30,
    "gpqa": 0.15,
    "mt_bench": 0.10,
    "ruler": 0.10,
}

assert math.isclose(sum(BENCHMARK_WEIGHTS.values()), 1.0), "benchmark weights must sum to 1.0"


@dataclass(frozen=True)
class ModelScore:
    """Composite scoring breakdown for a single model."""

    name: str
    benchmark_score: float
    tier_score: float
    context_window_score: float
    cost_score: float
    provider_multiplier: float
    strategy_score: float
    details: dict[str, Any]


def _lookup_benchmark_scores(name: str) -> dict[str, float] | None:
    """Return benchmark scores for a model using substring matching."""
    lowered = name.lower()
    for key, scores in BENCHMARK_SCORES.items():
        if key.lower() in lowered:
            return scores
    return None


def _benchmark_quality_score(scores: dict[str, float] | None) -> float:
    """Compute weighted normalized score from available benchmarks.

    Missing benchmarks are ignored; if none are available, returns 0.0 so the
    caller can fall back to the tier score.
    """
    if not scores:
        return 0.0

    available: dict[str, float] = {
        name: value
        for name, value in scores.items()
        if name in BENCHMARK_WEIGHTS and value > 0
    }
    if not available:
        return 0.0

    total_weight = sum(BENCHMARK_WEIGHTS[name] for name in available)
    weighted = sum(scores[name] * BENCHMARK_WEIGHTS[name] for name in available)
    return weighted / total_weight


def _tier_score(tier: Tier) -> float:
    """Convert tier enum to a normalized numeric score."""
    return tier.value / 3.0


def _context_window_score(context_window: int) -> float:
    """Normalize context window using a log scale capped at 1M tokens."""
    if context_window <= 0:
        return 0.0
    return min(1.0, math.log10(context_window) / math.log10(1_000_000))


def _cost_score(cost_ratio: float) -> float:
    """Convert cost ratio to a normalized score (cheaper = higher).

    Uses a sigmoidal decay so that very cheap models are close to 1.0 and
    expensive ones approach 0.0 smoothly.
    """
    if cost_ratio <= 0:
        return 1.0
    return 1.0 / (1.0 + cost_ratio / 10.0)


def _provider_multiplier(provider: Provider, provider_cost_order: list[str]) -> float:
    """Return a small cost tie-break multiplier based on provider ranking.

    The first provider in provider_cost_order gets 1.00, the second 0.97,
    third 0.94, and so on. Providers not in the list receive 0.70.
    """
    try:
        index = [p.lower() for p in provider_cost_order].index(provider.value.lower())
    except ValueError:
        return 0.70
    return max(0.40, 1.0 - index * 0.03)


def score_model(
    model: ModelInfo,
    *,
    strategy: str,
    provider_cost_order: list[str],
) -> ModelScore:
    """Compute a full scoring breakdown for a model under a routing strategy."""
    benchmark_scores = _lookup_benchmark_scores(model.name)
    benchmark_score = _benchmark_quality_score(benchmark_scores)

    tier_score = _tier_score(model.tier)
    ctx_score = _context_window_score(model.context_window)
    cost = _cost_score(model.cost_ratio)
    provider_mult = _provider_multiplier(model.provider, provider_cost_order)

    # If no benchmark data is known, fall back to tier as the primary signal.
    effective_benchmark = benchmark_score if benchmark_score > 0 else tier_score

    # Quality component: benchmark-first, but metadata smooths out the result.
    quality_score = (
        0.70 * effective_benchmark
        + 0.15 * tier_score
        + 0.10 * ctx_score
        + 0.05 * cost
    )

    strategy = strategy.lower()
    if strategy == "cost":
        strategy_score = (
            0.60 * cost
            + 0.25 * quality_score
            + 0.15 * provider_mult
        )
    elif strategy == "quality":
        strategy_score = quality_score
    elif strategy == "latency":
        # Prefer cheaper models (often smaller/faster) while keeping quality as tie-break.
        strategy_score = (
            0.55 * cost
            + 0.30 * quality_score
            + 0.15 * provider_mult
        )
    else:
        # balanced: penalize very expensive models, reward quality and provider ranking.
        cost_penalty = max(0.0, 1.0 - model.cost_ratio / 50.0)
        strategy_score = (
            0.55 * quality_score
            + 0.20 * ctx_score
            + 0.15 * provider_mult
            + 0.10 * cost_penalty
        )

    return ModelScore(
        name=model.name,
        benchmark_score=round(benchmark_score, 4),
        tier_score=round(tier_score, 4),
        context_window_score=round(ctx_score, 4),
        cost_score=round(cost, 4),
        provider_multiplier=round(provider_mult, 4),
        strategy_score=round(strategy_score, 4),
        details={
            "strategy": strategy,
            "provider_cost_order": provider_cost_order,
            "benchmark_breakdown": benchmark_scores,
        },
    )


def rank_models(
    models: list[ModelInfo],
    *,
    strategy: str,
    provider_cost_order: list[str],
) -> list[ModelInfo]:
    """Return models sorted by composite benchmark+metadata score descending."""
    scored = [
        (model, score_model(model, strategy=strategy, provider_cost_order=provider_cost_order))
        for model in models
    ]
    scored.sort(key=lambda item: item[1].strategy_score, reverse=True)
    return [model for model, _ in scored]
