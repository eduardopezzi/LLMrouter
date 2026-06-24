"""Evaluator domain types."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmrouter.core.types import RoutingGrade


@dataclass(frozen=True)
class QualityScore:
    """Quality dimensions judged on a 1-5 scale."""

    relevance: int
    accuracy: int
    completeness: int
    concision: int
    safety: int
    rationale: str = ""

    @property
    def overall(self) -> float:
        return (
            self.relevance
            + self.accuracy
            + self.completeness
            + self.concision
            + self.safety
        ) / 5.0


@dataclass(frozen=True)
class ComparisonResult:
    """A/B comparison result from the local judge."""

    winner: str
    confidence: float
    rationale: str = ""


@dataclass(frozen=True)
class RoutingObservation:
    """Data captured after a provider response returns to the caller."""

    prompt: str
    chosen_model: str
    response: str
    latency_ms: float
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    scorer_score: float | None = None
    scorer_tier: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingReview:
    """Annotated assessment of one routing observation."""

    observation_id: int
    quality: QualityScore
    grade: RoutingGrade
    suggested_model: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class TrainingExample:
    """Labeled scorer training example extracted from feedback data."""

    prompt: str
    chosen_model: str
    grade: RoutingGrade
    quality_overall: float
    scorer_score: float | None = None
    scorer_tier: int | None = None
