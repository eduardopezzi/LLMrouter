from __future__ import annotations

import httpx
import pytest

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, RoutingGrade, Tier
from llmrouter.evaluator.collector import ObservationCollector
from llmrouter.evaluator.grader import RoutingDecisionGrader
from llmrouter.evaluator.judge import QualityJudge
from llmrouter.evaluator.types import QualityScore, RoutingObservation, RoutingReview


@pytest.mark.asyncio
async def test_quality_judge_parses_ollama_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": """
                    {
                      "relevance": 5,
                      "accuracy": 4,
                      "completeness": 5,
                      "concision": 4,
                      "safety": 5,
                      "rationale": "solid"
                    }
                    """
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        judge = QualityJudge(client=client)
        score = await judge.evaluate("prompt", "response", "model")

    assert score.overall == pytest.approx(4.6)
    assert score.rationale == "solid"


@pytest.mark.asyncio
async def test_observation_collector_flushes_and_loads_training_data(tmp_path) -> None:
    collector = ObservationCollector(db_path=str(tmp_path / "feedback.db"))
    collector.record(
        RoutingObservation(
            prompt="Say hello",
            chosen_model="cheap",
            response="Hello",
            latency_ms=42,
            scorer_score=0.1,
            scorer_tier=1,
        )
    )

    ids = await collector.flush()
    assert len(ids) == 1

    await collector.save_review(
        RoutingReview(
            observation_id=ids[0],
            quality=QualityScore(5, 5, 5, 5, 5),
            grade=RoutingGrade.OPTIMAL,
        )
    )
    training_data = await collector.get_training_data()

    assert len(training_data) == 1
    assert training_data[0].prompt == "Say hello"
    assert training_data[0].grade == RoutingGrade.OPTIMAL


def test_routing_grader_marks_bad_quality_as_underkill() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),
            ModelInfo(name="strong", provider=Provider.OPENAI, tier=Tier.T3),
        )
    )
    grader = RoutingDecisionGrader()

    grade = grader.grade_decision(
        prompt="Explain a subtle distributed systems outage",
        chosen_model="cheap",
        response="No idea",
        quality_score=QualityScore(2, 2, 2, 3, 5),
        cost=0.001,
        latency_ms=100,
        registry=registry,
    )

    assert grade == RoutingGrade.UNDERKILL
