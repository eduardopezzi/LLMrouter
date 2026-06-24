"""Routing decision grader."""

from __future__ import annotations

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, RoutingGrade, Tier
from llmrouter.evaluator.types import QualityScore


class RoutingDecisionGrader:
    """Assess whether the router chose an appropriate model tier."""

    def grade_decision(
        self,
        prompt: str,
        chosen_model: str,
        response: str,
        quality_score: QualityScore,
        cost: float,
        latency_ms: float,
        registry: ModelRegistry | None = None,
    ) -> RoutingGrade:
        """Grade a routing decision using quality and efficiency signals."""
        model = registry.get(chosen_model) if registry is not None else None
        quality = quality_score.overall

        if quality <= 2.8:
            return RoutingGrade.UNDERKILL
        if model is not None and model.tier == Tier.T3 and quality >= 4.0 and _looks_simple(prompt):
            return RoutingGrade.OVERKILL
        if cost > 0.02 and quality < 4.0:
            return RoutingGrade.OVERKILL
        if latency_ms > 15000 and quality < 4.2:
            return RoutingGrade.CORRECT
        if quality >= 4.4:
            return RoutingGrade.OPTIMAL
        return RoutingGrade.CORRECT

    def suggest_alternative(
        self,
        prompt: str,
        chosen_model: str,
        registry: ModelRegistry,
    ) -> list[ModelInfo]:
        """Suggest cheaper or stronger alternatives from the registry."""
        chosen = registry.get(chosen_model)
        models = [model for model in registry.all() if model.name != chosen_model]
        if not models:
            return []

        if chosen is not None and chosen.tier == Tier.T3 and _looks_simple(prompt):
            cheaper = [model for model in models if model.tier.value < chosen.tier.value]
            return sorted(cheaper, key=lambda model: (model.tier.value, model.cost_ratio))[:3]

        return sorted(models, key=lambda model: (-model.tier.value, model.cost_ratio))[:3]


def _looks_simple(prompt: str) -> bool:
    words = prompt.split()
    hard_terms = {"architect", "debug", "derive", "optimize", "prove", "refactor", "review"}
    normalized_words = {word.lower().strip(".,:;!?") for word in words}
    return len(words) < 40 and not (normalized_words & hard_terms)
