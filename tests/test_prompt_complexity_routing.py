from __future__ import annotations

import pytest

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import ChatMessage, ChatRequest, ModelInfo, Provider, Tier


def test_simple_summary_is_classified_for_a_light_model() -> None:
    result = PromptScorer().score("Resuma esta frase em uma linha.")

    assert result.tier == Tier.T1
    assert result.signals["complexity_level"] == "simple"
    assert result.signals["task_type"] == "summarization"


def test_portuguese_security_audit_is_complex() -> None:
    result = PromptScorer().score(
        "Faça uma auditoria de segurança procurando vulnerabilidades OWASP e injeção SQL."
    )

    assert result.tier == Tier.T3
    assert result.signals["complexity_level"] == "complex"
    assert result.signals["task_type"] == "security_audit"


def test_multiple_requested_actions_raise_complexity() -> None:
    result = PromptScorer().score(
        "Analise, refatore, implemente e teste a solução antes de revisar o resultado."
    )

    assert result.tier == Tier.T3
    assert result.signals["complexity_floor"] >= 0.67


@pytest.mark.asyncio
async def test_inferred_task_redirects_to_specialist_in_another_tier() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="generic-medium",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=1,
            ),
            ModelInfo(
                name="review-specialist",
                provider=Provider.OLLAMA,
                tier=Tier.T3,
                priority=50,
                capabilities=frozenset({"review"}),
            ),
        )
    )
    router = MultiModelRouter(registry, PromptScorer(), strategy="quality")

    decision = await router.route(
        ChatRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Revise este pull request curto.")]
        )
    )

    assert decision.tier == Tier.T2
    assert decision.primary.name == "review-specialist"
    assert "task=review" in decision.reason


@pytest.mark.asyncio
async def test_simple_prompt_keeps_light_tier() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="light-summary",
                provider=Provider.OLLAMA,
                tier=Tier.T1,
                capabilities=frozenset({"summarization"}),
            ),
            ModelInfo(
                name="large-summary",
                provider=Provider.OLLAMA,
                tier=Tier.T3,
                capabilities=frozenset({"summarization"}),
            ),
        )
    )
    router = MultiModelRouter(registry, PromptScorer(), strategy="quality")

    decision = await router.route(
        ChatRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Resuma esta frase.")],
        )
    )

    assert decision.tier == Tier.T1
    assert decision.primary.name == "light-summary"
