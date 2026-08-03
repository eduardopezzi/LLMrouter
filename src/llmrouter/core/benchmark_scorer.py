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

# Kept as an empty compatibility hook. Scores must come from an explicit
# ``benchmark_scores`` catalog entry or the versioned refresh catalog; the
# router never ships estimated/illustrative performance figures.
BENCHMARK_SCORES: dict[str, dict[str, float]] = {}

# Which benchmarks participate in the composite quality score and with what weight.
BENCHMARK_WEIGHTS: dict[str, float] = {
    "mmlu": 0.35,
    "humaneval": 0.30,
    "gpqa": 0.15,
    "mt_bench": 0.10,
    "ruler": 0.10,
}

assert math.isclose(sum(BENCHMARK_WEIGHTS.values()), 1.0), "benchmark weights must sum to 1.0"

_BENCHMARK_ALIASES: dict[str, str] = {
    "mmlu": "MMLU-Pro",
    "mmlupro": "MMLU-Pro",
    "simpleqa": "SimpleQA",
    "gpqa": "GPQA Diamond",
    "gpqadiamond": "GPQA Diamond",
    "hle": "Humanity's Last Exam (HLE)",
    "humanityslastexam": "Humanity's Last Exam (HLE)",
    "hmmt": "HMMT",
    "imoanswerbench": "IMOAnswerBench",
    "math": "MATH",
    "gsm8k": "GSM8K",
    "livecodebench": "LiveCodeBench",
    "codeforces": "Codeforces",
    "mrcr": "MRCR 1M",
    "mrcr1m": "MRCR 1M",
    "corpusqa": "CorpusQA 1M",
    "corpusqa1m": "CorpusQA 1M",
    "terminalbench20": "TerminalBench 2.0",
    "swebenchverified": "SWE-Bench Verified",
    "swebenchpro": "SWE-Bench Pro",
    "browsecomp": "BrowseComp",
    "ifeval": "IFEval",
    "bfcl": "BFCL",
    "musr": "MuSR",
}


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


def _benchmark_quality_score(
    scores: dict[str, float] | None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted normalized score from available benchmarks.

    Missing benchmarks are ignored; if none are available, returns 0.0 so the
    caller can fall back to the tier score.
    """
    if not scores:
        return 0.0

    active_weights = weights or BENCHMARK_WEIGHTS
    normalized_weights = {
        _canonical_benchmark_name(name): value
        for name, value in active_weights.items()
        if value > 0
    }
    normalized_scores = {
        _canonical_benchmark_name(name): _normalize_benchmark_value(name, value)
        for name, value in scores.items()
        if value > 0
    }
    available = {
        name: value for name, value in normalized_scores.items() if name in normalized_weights
    }
    if not available:
        return 0.0

    total_weight = sum(normalized_weights[name] for name in available)
    weighted = sum(available[name] * normalized_weights[name] for name in available)
    return weighted / total_weight


def _canonical_benchmark_name(name: str) -> str:
    compact = "".join(character for character in name.lower() if character.isalnum())
    return _BENCHMARK_ALIASES.get(compact, name.strip())


def _normalize_benchmark_value(name: str, value: float) -> float:
    """Normalize percentage, 0–1, and native Codeforces rating values."""
    numeric = float(value)
    if _canonical_benchmark_name(name) == "Codeforces" and numeric > 100:
        return min(1.0, max(0.0, (numeric - 800.0) / (4000.0 - 800.0)))
    if numeric > 1.0:
        numeric /= 100.0
    return min(1.0, max(0.0, numeric))


def _benchmark_coverage(
    scores: dict[str, float] | None,
    weights: dict[str, float] | None,
) -> float:
    if not scores or not weights:
        return 0.0
    available = {_canonical_benchmark_name(name) for name in scores}
    normalized_weights = {
        _canonical_benchmark_name(name): value for name, value in weights.items() if value > 0
    }
    return sum(value for name, value in normalized_weights.items() if name in available)


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
    benchmark_weights: dict[str, float] | None = None,
) -> ModelScore:
    """Compute a full scoring breakdown for a model under a routing strategy."""
    benchmark_scores = dict(model.benchmark_scores) or _lookup_benchmark_scores(model.name)
    benchmark_score = _benchmark_quality_score(benchmark_scores, benchmark_weights)
    benchmark_coverage = _benchmark_coverage(benchmark_scores, benchmark_weights)

    tier_score = _tier_score(model.tier)
    ctx_score = _context_window_score(model.context_window)
    cost = _cost_score(model.cost_ratio)
    provider_mult = _provider_multiplier(model.provider, provider_cost_order)

    # If no benchmark data is known, fall back to tier as the primary signal.
    effective_benchmark = benchmark_score if benchmark_score > 0 else tier_score
    if benchmark_weights is not None and benchmark_coverage == 0:
        effective_benchmark = 0.5 * tier_score

    # Quality component: benchmark-first, but metadata smooths out the result.
    quality_score = 0.70 * effective_benchmark + 0.15 * tier_score + 0.10 * ctx_score + 0.05 * cost

    strategy = strategy.lower()
    if strategy == "cost":
        strategy_score = 0.60 * cost + 0.25 * quality_score + 0.15 * provider_mult
    elif strategy == "quality":
        strategy_score = quality_score
    elif strategy == "latency":
        # Prefer cheaper models (often smaller/faster) while keeping quality as tie-break.
        strategy_score = 0.55 * cost + 0.30 * quality_score + 0.15 * provider_mult
    else:
        # balanced: penalize very expensive models, reward quality and provider ranking.
        cost_penalty = max(0.0, 1.0 - model.cost_ratio / 50.0)
        strategy_score = (
            0.55 * quality_score + 0.20 * ctx_score + 0.15 * provider_mult + 0.10 * cost_penalty
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
            "benchmark_weights": benchmark_weights,
            "benchmark_coverage": round(benchmark_coverage, 4),
        },
    )


def rank_models(
    models: list[ModelInfo],
    *,
    strategy: str,
    provider_cost_order: list[str],
    benchmark_weights: dict[str, float] | None = None,
) -> list[ModelInfo]:
    """Return models sorted by composite benchmark+metadata score descending."""
    scored = [
        (
            model,
            score_model(
                model,
                strategy=strategy,
                provider_cost_order=provider_cost_order,
                benchmark_weights=benchmark_weights,
            ),
        )
        for model in models
    ]
    scored.sort(key=lambda item: item[1].strategy_score, reverse=True)
    return [model for model, _ in scored]
