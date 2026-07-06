"""Tests for evaluator judge, grader, and synthetic data generator."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, RoutingGrade, Tier
from llmrouter.evaluator.grader import RoutingDecisionGrader, _looks_simple
from llmrouter.evaluator.judge import (
    QualityJudge,
    _clamp_score,
    _extract_json,
    _ollama_content,
)
from llmrouter.evaluator.synthetic import SyntheticDataGenerator
from llmrouter.evaluator.types import ComparisonResult, QualityScore


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------


def test_clamp_score_valid() -> None:
    assert _clamp_score(3) == 3
    assert _clamp_score(1) == 1
    assert _clamp_score(5) == 5


def test_clamp_score_out_of_range() -> None:
    assert _clamp_score(0) == 1
    assert _clamp_score(10) == 5
    assert _clamp_score(-5) == 1


def test_clamp_score_invalid() -> None:
    assert _clamp_score(None) == 3
    assert _clamp_score("abc") == 3
    assert _clamp_score(3.7) == 3


def test_extract_json_plain() -> None:
    result = _extract_json('{"a": 1}')
    assert result == {"a": 1}


def test_extract_json_with_whitespace() -> None:
    result = _extract_json('  {"a": 1}  ')
    assert result == {"a": 1}


def test_extract_json_fenced() -> None:
    result = _extract_json('```json\n{"a": 1}\n```')
    assert result == {"a": 1}


def test_extract_json_fenced_no_lang() -> None:
    result = _extract_json('```\n{"a": 1}\n```')
    assert result == {"a": 1}


def test_extract_json_with_prefix_text() -> None:
    result = _extract_json('Here is the result: {"a": 1} done.')
    assert result == {"a": 1}


def test_extract_json_not_dict() -> None:
    with pytest.raises(ValueError):
        _extract_json('[1, 2, 3]')


def test_ollama_content_with_message() -> None:
    body = {"message": {"content": "hello"}}
    assert _ollama_content(body) == "hello"


def test_ollama_content_with_response() -> None:
    body = {"response": "world"}
    assert _ollama_content(body) == "world"


def test_ollama_content_missing() -> None:
    with pytest.raises(ValueError):
        _ollama_content({})


# ---------------------------------------------------------------------------
# QualityJudge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_evaluate_success() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())
    response_body = {
        "message": {
            "content": json.dumps({
                "relevance": 4,
                "accuracy": 5,
                "completeness": 3,
                "concision": 4,
                "safety": 5,
                "rationale": "good",
            })
        }
    }

    original_post = judge._client.post  # type: ignore[union-attr]

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=response_body,
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    quality = await judge.evaluate("prompt", "response", "model")
    assert quality.relevance == 4
    assert quality.accuracy == 5
    assert quality.overall == pytest.approx((4 + 5 + 3 + 4 + 5) / 5.0)
    assert quality.rationale == "good"
    await judge._client.aclose()


@pytest.mark.asyncio
async def test_judge_compare_success() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"message": {"content": json.dumps({"winner": "a", "confidence": 0.8, "rationale": "a better"})}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    result = await judge.compare("prompt", "resp_a", "resp_b")
    assert result.winner == "a"
    assert result.confidence == pytest.approx(0.8)
    assert result.rationale == "a better"
    await judge._client.aclose()


@pytest.mark.asyncio
async def test_judge_compare_invalid_winner_defaults_to_tie() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"message": {"content": json.dumps({"winner": "invalid", "confidence": "bad"})}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    result = await judge.compare("p", "a", "b")
    assert result.winner == "tie"
    assert result.confidence == pytest.approx(0.5)
    await judge._client.aclose()


def test_judge_headers() -> None:
    judge = QualityJudge(api_key="test-key")
    headers = judge._headers()
    assert headers["Authorization"] == "Bearer test-key"

    judge_no_key = QualityJudge()
    assert judge_no_key._headers() == {}


# ---------------------------------------------------------------------------
# RoutingDecisionGrader
# ---------------------------------------------------------------------------


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1, cost_per_1k_input=0.001),
            ModelInfo(name="powerful", provider=Provider.OPENAI, tier=Tier.T3, cost_per_1k_input=0.1),
        )
    )


def test_grader_underkill_low_quality() -> None:
    grader = RoutingDecisionGrader()
    quality = QualityScore(1, 1, 1, 1, 1)
    grade = grader.grade_decision("prompt", "cheap", "response", quality, cost=0.001, latency_ms=100)
    assert grade == RoutingGrade.UNDERKILL


def test_grader_overkill_t3_simple_prompt() -> None:
    grader = RoutingDecisionGrader()
    quality = QualityScore(5, 5, 5, 5, 5)
    grade = grader.grade_decision(
        "hello world", "powerful", "ok", quality, cost=0.1, latency_ms=100,
        registry=_registry(),
    )
    assert grade == RoutingGrade.OVERKILL


def test_grader_overkill_high_cost_low_quality() -> None:
    grader = RoutingDecisionGrader()
    quality = QualityScore(3, 3, 3, 3, 3)
    grade = grader.grade_decision("prompt", "cheap", "ok", quality, cost=0.05, latency_ms=100)
    assert grade == RoutingGrade.OVERKILL


def test_grader_correct_high_latency() -> None:
    grader = RoutingDecisionGrader()
    quality = QualityScore(4, 4, 4, 4, 4)
    grade = grader.grade_decision("prompt", "cheap", "ok", quality, cost=0.001, latency_ms=20000)
    assert grade == RoutingGrade.CORRECT


def test_grader_optimal_high_quality() -> None:
    grader = RoutingDecisionGrader()
    quality = QualityScore(5, 5, 5, 5, 5)
    grade = grader.grade_decision("prompt", "cheap", "ok", quality, cost=0.001, latency_ms=100)
    assert grade == RoutingGrade.OPTIMAL


def test_grader_correct_default() -> None:
    grader = RoutingDecisionGrader()
    quality = QualityScore(4, 4, 4, 4, 4)
    grade = grader.grade_decision("prompt", "cheap", "ok", quality, cost=0.001, latency_ms=100)
    assert grade == RoutingGrade.CORRECT


def test_grader_suggest_alternative_overkill() -> None:
    grader = RoutingDecisionGrader()
    reg = _registry()
    alternatives = grader.suggest_alternative("hello world", "powerful", reg)
    assert len(alternatives) == 1
    assert alternatives[0].name == "cheap"


def test_grader_suggest_alternative_default() -> None:
    grader = RoutingDecisionGrader()
    reg = _registry()
    alternatives = grader.suggest_alternative("complex architecture design", "cheap", reg)
    assert len(alternatives) == 1
    assert alternatives[0].name == "powerful"


def test_grader_suggest_alternative_empty_registry() -> None:
    grader = RoutingDecisionGrader()
    reg = ModelRegistry()
    alternatives = grader.suggest_alternative("prompt", "model", reg)
    assert alternatives == []


def test_looks_simple_short_no_hard_terms() -> None:
    assert _looks_simple("hello world") is True


def test_looks_simple_short_with_hard_terms() -> None:
    assert _looks_simple("please debug this issue") is False


def test_looks_simple_long() -> None:
    assert _looks_simple(" ".join(["word"] * 50)) is False


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthetic_generate_prompts() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"message": {"content": json.dumps({"prompts": ["p1", "p2", "p3"]})}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    generator = SyntheticDataGenerator(judge)
    prompts = await generator.generate_prompts("simple", count=3)
    assert prompts == ["p1", "p2", "p3"]
    await judge._client.aclose()


@pytest.mark.asyncio
async def test_synthetic_generate_adversarial_prompts() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"message": {"content": json.dumps({"prompts": ["adv1", "adv2"]})}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    generator = SyntheticDataGenerator(judge)
    prompts = await generator.generate_adversarial_prompts(count=2)
    assert prompts == ["adv1", "adv2"]
    await judge._client.aclose()


@pytest.mark.asyncio
async def test_synthetic_generate_prompts_string_response() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"message": {"content": json.dumps({"prompts": "single prompt string"})}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    generator = SyntheticDataGenerator(judge)
    prompts = await generator.generate_prompts("simple", count=1)
    assert prompts == ["single prompt string"]
    await judge._client.aclose()


@pytest.mark.asyncio
async def test_synthetic_generate_prompts_not_list() -> None:
    judge = QualityJudge(client=httpx.AsyncClient())

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"message": {"content": json.dumps({"prompts": 123})}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    judge._client.post = mock_post  # type: ignore[assignment,union-attr]
    generator = SyntheticDataGenerator(judge)
    prompts = await generator.generate_prompts("simple", count=1)
    assert prompts == []
    await judge._client.aclose()