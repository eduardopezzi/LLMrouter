"""Feedback loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import RoutingGrade
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.grader import RoutingDecisionGrader
from llmrouter.evaluator.judge import QualityJudge
from llmrouter.evaluator.types import RoutingReview


@dataclass(frozen=True)
class FeedbackReport:
    """Summary of one feedback cycle."""

    evaluated: int
    optimal: int
    correct: int
    overkill: int
    underkill: int


class FeedbackLoop:
    """Runs background evaluation cycles for pending observations."""

    def __init__(
        self,
        collector: ObservationCollector,
        judge: QualityJudge,
        grader: RoutingDecisionGrader,
        registry: ModelRegistry,
    ) -> None:
        self._collector = collector
        self._judge = judge
        self._grader = grader
        self._registry = registry

    async def run_cycle(self, limit: int = 50) -> FeedbackReport:
        """Flush observations, evaluate pending rows, and persist reviews."""
        await self._collector.flush()
        pending = await self._collector.get_pending_observations(limit=limit)
        counts = {
            RoutingGrade.OPTIMAL: 0,
            RoutingGrade.CORRECT: 0,
            RoutingGrade.OVERKILL: 0,
            RoutingGrade.UNDERKILL: 0,
        }

        for observation_id, observation in pending:
            quality = await self._judge.evaluate(
                observation.prompt,
                observation.response,
                observation.chosen_model,
            )
            grade = self._grader.grade_decision(
                prompt=observation.prompt,
                chosen_model=observation.chosen_model,
                response=observation.response,
                quality_score=quality,
                cost=observation.cost_usd,
                latency_ms=observation.latency_ms,
                registry=self._registry,
            )
            alternatives = self._grader.suggest_alternative(
                observation.prompt,
                observation.chosen_model,
                self._registry,
            )
            await self._collector.save_review(
                RoutingReview(
                    observation_id=observation_id,
                    quality=quality,
                    grade=grade,
                    suggested_model=alternatives[0].name if alternatives else None,
                )
            )
            counts[grade] += 1

        return FeedbackReport(
            evaluated=len(pending),
            optimal=counts[RoutingGrade.OPTIMAL],
            correct=counts[RoutingGrade.CORRECT],
            overkill=counts[RoutingGrade.OVERKILL],
            underkill=counts[RoutingGrade.UNDERKILL],
        )
