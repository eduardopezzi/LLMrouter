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

from typing import Protocol

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.scorer import PromptScorer, ScoringResult
from llmrouter.core.types import (
    ChatRequest,
    ModelInfo,
    RoutingConstraints,
    RoutingDecision,
    RoutingStrategy,
    Tier,
)
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.router")


class SelectionStrategy(Protocol):
    """Protocol for tier-internal model selection strategies."""

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        """Order models by preference. Returns the full sorted list."""
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class CostStrategy:
    """Prefer the cheapest model within the tier."""

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        return sorted(models, key=lambda m: m.cost_ratio)


class QualityStrategy:
    """Prefer the highest-tier / most capable model.

    Within a tier, uses lower priority number as a proxy for higher quality.
    """

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        return sorted(models, key=lambda m: (m.priority, -m.max_tokens))


class LatencyStrategy:
    """Prefer models that are likely faster (lower cost ratio as proxy).

    Local models (Ollama) are preferred when available since they have no
    network latency.
    """

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        return sorted(models, key=lambda m: (m.provider != "ollama", m.cost_ratio))


class BalancedStrategy:
    """Balance cost and quality using a composite score.

    Score = normalized_cost * 0.5 + normalized_priority * 0.5
    Lower score is better.
    """

    def select(self, models: list[ModelInfo], constraints: RoutingConstraints) -> list[ModelInfo]:
        if not models:
            return []

        max_cost = max(m.cost_ratio for m in models) or 1.0
        return sorted(
            models,
            key=lambda m: (m.cost_ratio / max_cost) * 0.5 + (m.priority / 10.0) * 0.5,
        )


_STRATEGY_MAP: dict[RoutingStrategy, type[SelectionStrategy]] = {
    RoutingStrategy.COST: CostStrategy,
    RoutingStrategy.QUALITY: QualityStrategy,
    RoutingStrategy.LATENCY: LatencyStrategy,
    RoutingStrategy.BALANCED: BalancedStrategy,
}


def get_strategy(strategy: RoutingStrategy) -> SelectionStrategy:
    """Return the selection strategy implementation for the given enum value."""
    cls = _STRATEGY_MAP.get(strategy, BalancedStrategy)
    return cls()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


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
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        fallback_count: int = 2,
    ) -> None:
        self._registry = registry
        self._scorer = scorer
        self._strategy = get_strategy(strategy)
        self._fallback_count = fallback_count

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
            candidates = self._registry.all()
            _logger.debug("No candidates in tier %s, using all %d models", scoring.tier.name, len(candidates))

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
        ordered = self._strategy.select(candidates, constraints)
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
        candidates = self._registry.by_tier(tier)

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
                candidates = self._registry.by_tier(fallback_tier)
                if candidates:
                    break

        # Filter by required capabilities
        if constraints.required_capabilities:
            candidates = [
                m for m in candidates if constraints.required_capabilities <= m.capabilities
            ]

        return candidates

    def _build_fallbacks(
        self,
        primary: ModelInfo,
        constraints: RoutingConstraints,
    ) -> list[ModelInfo]:
        """Build fallback chain excluding the primary model."""
        all_models = self._registry.all()
        if primary in all_models:
            all_models = [m for m in all_models if m.name != primary.name]
        ordered = self._strategy.select(all_models, constraints)
        return ordered[: self._fallback_count]

    @staticmethod
    def _build_reason(scoring: ScoringResult, model: ModelInfo) -> str:
        """Build a human-readable reason string."""
        top_signals = sorted(scoring.signals.items(), key=lambda x: x[1], reverse=True)[:2]
        signal_str = ", ".join(f"{name}={val:.2f}" for name, val in top_signals)
        return (
            f"Prompt scored {scoring.score:.2f} (tier {scoring.tier.name}), "
            f"signals: {signal_str}. Selected {model.name} via {model.provider.value}."
        )
