"""Multi-model router — selects the best model for a given request.

The router combines the :class:`PromptScorer` (complexity assessment) with
the :class:`ModelRegistry` (available models) to produce a
:class:`RoutingDecision` containing the primary model and ordered fallbacks.

Design goals:
- Functional style: pure functions where possible, no hidden state.
- Strategy pattern: pluggable selection strategy (cost, quality, balanced, latency).
- Extensible: the scorer can be swapped for an ML-based one without changing
  the router interface.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from llmrouter.core.health import HealthScore, ModelHealthTracker
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.scorer import PromptScorer, ScoringResult
from llmrouter.core.types import (
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingConstraints,
    RoutingDecision,
    RoutingStrategy,
    Tier,
)
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.router")


def asyncio_run(coro: Any) -> Any:
    """Run an async coroutine regardless of whether an event loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


class SelectionStrategy(Protocol):
    """Protocol for tier-internal model selection strategies."""

    def set_health_tracker(self, tracker: ModelHealthTracker | None) -> None:
        """Optionally wire a health tracker for adaptive routing."""
        ...

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        """Order models by preference. Returns the full sorted list."""
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

_PROVIDER_COST_RANK = {
    Provider.NVIDIA: 0,
    Provider.ZAI: 1,
    Provider.OLLAMA: 2,
}


def asyncio_run(coro: Any) -> Any:
    """Run an async coroutine regardless of whether an event loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # If we are inside a running loop (e.g. pytest-asyncio), create a new
    # event loop in a dedicated thread to avoid blocking the current one.
    if loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=5)
    return loop.run_until_complete(coro)


def _health_scores_from_tracker(
    tracker: ModelHealthTracker | None, models: list[ModelInfo]
) -> dict[str, float]:
    """Load composite health scores for the given models."""
    if tracker is None:
        return {}

    async def _fetch() -> dict[str, float]:
        score_map = await tracker.score_map()
        return {name: score.score for name, score in score_map.items()}

    return asyncio_run(_fetch())


class CostStrategy:
    """Prefer the cheapest model within the tier.

    Numeric model costs are primary. When the catalog has equal or zero costs,
    providers are ranked by the current commercial preference:
    NVIDIA, then Zhipu, then Ollama.

    When a health tracker is wired, the composite HealthScore (0-1, higher is
    better) is used to demote sick models before the final priority tie-break.
    """

    def __init__(self, provider_cost_order: list[str] | None = None) -> None:
        self._provider_cost_rank = _provider_cost_rank(provider_cost_order)
        self._health_tracker: ModelHealthTracker | None = None

    def set_health_tracker(self, tracker: ModelHealthTracker | None) -> None:
        self._health_tracker = tracker

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        scores = _health_scores_from_tracker(self._health_tracker, models)
        return sorted(
            models,
            key=lambda m: (
                m.cost_ratio,
                -scores.get(m.name, 1.0),  # higher score is better
                self._provider_cost_rank.get(m.provider, 99),
                m.priority,
            ),
        )


class QualityStrategy:
    """Prefer the highest-tier / most capable model.

    Within a tier, uses lower priority number as a proxy for higher quality.
    When a health tracker is wired, healthier models are preferred between
    models with the same catalog priority.
    """

    def __init__(self) -> None:
        self._health_tracker: ModelHealthTracker | None = None

    def set_health_tracker(self, tracker: ModelHealthTracker | None) -> None:
        self._health_tracker = tracker

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        scores = _health_scores_from_tracker(self._health_tracker, models)
        return sorted(
            models,
            key=lambda m: (
                m.priority,
                -scores.get(m.name, 1.0),
                -m.max_tokens,
            ),
        )


class LatencyStrategy:
    """Prefer models that are likely faster (lower cost ratio as proxy).

    Local models (Ollama) are preferred when available since they have no
    network latency. When a health tracker is wired, real P95 latency is used
    as a tie-break instead of the static cost ratio.
    """

    def __init__(self) -> None:
        self._health_tracker: ModelHealthTracker | None = None

    def set_health_tracker(self, tracker: ModelHealthTracker | None) -> None:
        self._health_tracker = tracker

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        scores = _health_scores_from_tracker(self._health_tracker, models)
        # Prefer real P95 from health tracker when available
        health_map = self._health_p95_map(models)
        return sorted(
            models,
            key=lambda m: (
                health_map.get(m.name, float("inf")),
                m.provider != Provider.OLLAMA,
                -scores.get(m.name, 1.0),
                m.cost_ratio,
            ),
        )

    def _health_p95_map(self, models: list[ModelInfo]) -> dict[str, float]:
        if self._health_tracker is None:
            return {}

        async def _fetch() -> dict[str, float]:
            health_list = await self._health_tracker.list_health()
            return {h.model_name: h.p95_ms for h in health_list if h.p95_ms > 0}

        return asyncio_run(_fetch())


class BalancedStrategy:
    """Balance cost and quality using a composite score.

    Score = normalized_cost * 0.5 + normalized_priority * 0.5
    Lower score is better. When a health tracker is wired, priority is
    replaced by (1 - HealthScore) so unhealthy models are penalized.
    """

    def __init__(self) -> None:
        self._health_tracker: ModelHealthTracker | None = None

    def set_health_tracker(self, tracker: ModelHealthTracker | None) -> None:
        self._health_tracker = tracker

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        if not models:
            return []

        scores = _health_scores_from_tracker(self._health_tracker, models)
        max_cost = max(m.cost_ratio for m in models) or 1.0
        return sorted(
            models,
            key=lambda m: (
                (m.cost_ratio / max_cost) * 0.5
                + ((1.0 - scores.get(m.name, 1.0)) * 0.25)
                + (m.priority / 10.0) * 0.25
            ),
        )


_STRATEGY_MAP: dict[RoutingStrategy, type[SelectionStrategy]] = {
    RoutingStrategy.COST: CostStrategy,
    RoutingStrategy.QUALITY: QualityStrategy,
    RoutingStrategy.LATENCY: LatencyStrategy,
    RoutingStrategy.BALANCED: BalancedStrategy,
}


def get_strategy(
    strategy: RoutingStrategy,
    provider_cost_order: list[str] | None = None,
) -> SelectionStrategy:
    """Return the selection strategy implementation for the given enum value."""
    cls = _STRATEGY_MAP.get(strategy, BalancedStrategy)
    if cls is CostStrategy:
        return cls(provider_cost_order)
    return cls()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


async def health_tracker_scores(
    tracker: ModelHealthTracker, models: list[ModelInfo]
) -> dict[str, HealthScore]:
    """Convenience async helper to load health scores for a candidate list."""
    all_scores = await tracker.score_map()
    return {
        m.name: all_scores.get(m.name, HealthScore(m.name, 1.0, 1.0, 1.0, 0.5, 1.0, 0))
        for m in models
    }


class MultiModelRouter:
    """Routes requests to the best available model.

    Workflow:
    1. Score the prompt complexity → determines the tier.
    2. Get all models in that tier (and higher tiers as fallbacks).
    3. Apply constraints (max cost, required capabilities).
    4. Use the selection strategy to order models within the tier.
    5. Build fallback chain from remaining models + adjacent tiers.

    The router is stateless (uses injected scorer and registry) and safe
    to share across async tasks.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        scorer: PromptScorer,
        strategy: RoutingStrategy = RoutingStrategy.COST,
        fallback_count: int = 2,
        provider_cost_order: list[str] | None = None,
        health_tracker: ModelHealthTracker | None = None,
    ) -> None:
        self._registry = registry
        self._scorer = scorer
        self._strategy = get_strategy(strategy, provider_cost_order)
        self._fallback_count = fallback_count
        self._unavailable_providers: set[Provider] = set()
        self._health_tracker = health_tracker
        if health_tracker is not None:
            self._strategy.set_health_tracker(health_tracker)

    def replace_registry(self, registry: ModelRegistry) -> None:
        """Replace the live model registry used for future routing decisions."""
        self._registry = registry

    def mark_provider_unavailable(self, provider: Provider) -> None:
        """Exclude a provider from future automatic routing decisions."""
        self._unavailable_providers.add(provider)

    def set_health_tracker(self, tracker: ModelHealthTracker | None) -> None:
        """Wire the health tracker into the selection strategy for adaptive routing."""
        self._health_tracker = tracker
        self._strategy.set_health_tracker(tracker)

    async def route(
        self,
        request: ChatRequest,
        constraints: RoutingConstraints | None = None,
    ) -> RoutingDecision:
        """Route a chat request to the best model.

        Args:
            request: The incoming chat request.
            constraints: Optional routing constraints (cost, capabilities).

        Returns:
            A :class:`RoutingDecision` with primary model + fallbacks.
        """
        constraints = constraints or RoutingConstraints()

        # If the request specifies a model, use it directly
        if request.model and request.model in self._registry:
            primary = self._registry.get(request.model)
            assert primary is not None
            fallbacks = self._build_fallbacks(primary, constraints)
            _logger.debug(
                "Explicit model selection: %s | fallbacks=%s",
                primary.name,
                [m.name for m in fallbacks] or "none",
            )
            return RoutingDecision(
                primary=primary,
                fallbacks=fallbacks,
                score=0.0,
                tier=primary.tier,
                reason=f"Explicit model selection: {request.model}",
            )

        # Score the prompt
        prompt_text = request.prompt_text
        scoring = self._scorer.score(prompt_text)

        # Debug: log scoring details
        _logger.debug(
            "Scoring: score=%.2f tier=%s | signals: %s",
            scoring.score,
            scoring.tier.name,
            ", ".join(f"{k}={v:.2f}" for k, v in sorted(
                scoring.signals.items(), key=lambda x: x[1], reverse=True
            )) or "none",
        )

        # Get candidate models for the recommended tier
        candidates = self._get_candidates(scoring.tier, constraints)

        if not candidates:
            # Fallback: try any model available
            candidates = self._available_models(self._registry.all())
            _logger.debug(
                "No candidates in tier %s, using all %d models",
                scoring.tier.name,
                len(candidates),
            )

        if not candidates:
            raise RuntimeError("No models available for routing")

        # Debug: log candidates
        _logger.debug(
            "Candidates (%d in tier %s): %s",
            len(candidates),
            scoring.tier.name,
            [m.name for m in candidates],
        )

        # Apply selection strategy
        ordered = _unique_models(self._strategy.select(candidates, constraints))
        primary = ordered[0]
        fallbacks = ordered[1 : 1 + self._fallback_count]

        # Debug: log final selection
        _logger.debug(
            "Strategy=%s → primary=%s (priority=%d) | fallbacks=%s",
            type(self._strategy).__name__,
            primary.name,
            primary.priority,
            [m.name for m in fallbacks] or "none",
        )

        return RoutingDecision(
            primary=primary,
            fallbacks=fallbacks,
            score=scoring.score,
            tier=scoring.tier,
            reason=self._build_reason(scoring, primary),
        )

    def _get_candidates(
        self, tier: Tier, constraints: RoutingConstraints
    ) -> list[ModelInfo]:
        """Get candidate models for a tier, filtered by constraints."""
        # Start with models in the recommended tier
        candidates = self._available_models(self._registry.by_tier(tier))

        # If no models in tier, try adjacent tiers
        if not candidates:
            fallback_tiers = [
                candidate_tier
                for candidate_tier in Tier
                if candidate_tier != tier
            ]
            fallback_tiers.sort(
                key=lambda candidate_tier: (
                    candidate_tier.value < tier.value,
                    abs(candidate_tier.value - tier.value),
                )
            )
            for fallback_tier in fallback_tiers:
                candidates = self._available_models(self._registry.by_tier(fallback_tier))
                if candidates:
                    break

        # Filter by required capabilities
        if constraints.required_capabilities:
            candidates = [
                m for m in candidates if constraints.required_capabilities <= m.capabilities
            ]
            if not candidates:
                candidates = [
                    m for m in self._registry.all()
                    if constraints.required_capabilities <= m.capabilities
                ]
                candidates = self._available_models(candidates)

        return candidates

    def _build_fallbacks(
        self,
        primary: ModelInfo,
        constraints: RoutingConstraints,
    ) -> list[ModelInfo]:
        """Build fallback chain excluding the primary model."""
        all_models = self._available_models(self._registry.all())
        if primary in all_models:
            all_models = [m for m in all_models if m.name != primary.name]
        ordered = _unique_models(self._strategy.select(all_models, constraints))
        return ordered[: self._fallback_count]

    def _available_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        if not self._unavailable_providers:
            return models
        return [model for model in models if model.provider not in self._unavailable_providers]

    @staticmethod
    def _build_reason(scoring: ScoringResult, model: ModelInfo) -> str:
        """Build a human-readable reason string."""
        top_signals = sorted(scoring.signals.items(), key=lambda x: x[1], reverse=True)[:2]
        signal_str = ", ".join(f"{name}={val:.2f}" for name, val in top_signals)
        return (
            f"Prompt scored {scoring.score:.2f} (tier {scoring.tier.name}), "
            f"signals: {signal_str}. Selected {model.name} via {model.provider.value}."
        )


def _provider_cost_rank(provider_cost_order: list[str] | None) -> dict[Provider, int]:
    if not provider_cost_order:
        return _PROVIDER_COST_RANK
    result: dict[Provider, int] = {}
    for index, provider_name in enumerate(provider_cost_order):
        try:
            result[Provider(provider_name)] = index
        except ValueError:
            continue
    for provider, rank in _PROVIDER_COST_RANK.items():
        result.setdefault(provider, len(result) + rank)
    return result


def _unique_models(models: list[ModelInfo]) -> list[ModelInfo]:
    unique: list[ModelInfo] = []
    seen: set[str] = set()
    for model in models:
        if model.name in seen:
            continue
        unique.append(model)
        seen.add(model.name)
    return unique
