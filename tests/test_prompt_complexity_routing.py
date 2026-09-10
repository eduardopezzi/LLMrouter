from __future__ import annotations

from threading import Event

import pytest

from llmrouter.core.registry import ModelRegistry
from llmrouter.core.router import MultiModelRouter, _inferred_task_type
from llmrouter.core.scorer import PromptScorer, ScoringResult
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


def test_long_transcript_without_complex_intent_does_not_force_t3() -> None:
    result = PromptScorer().score("contexto antigo " * 2_000 + "resuma esta frase")

    assert result.tier == Tier.T2
    assert result.signals["complexity_floor"] == pytest.approx(0.36)


def test_unreliable_semantic_role_does_not_select_specialist() -> None:
    result = ScoringResult(
        score=0.54,
        tier=Tier.T1,
        signals={
            "task_type": "general",
            "semantic_role": "review",
            "semantic_confidence": 0.54,
            "semantic_reliable": False,
        },
    )

    assert _inferred_task_type(result) is None


@pytest.mark.asyncio
async def test_router_scores_only_bounded_current_context() -> None:
    captured: list[str] = []

    class CaptureScorer:
        def score(self, prompt: str) -> ScoringResult:
            captured.append(prompt)
            return ScoringResult(score=0.1, tier=Tier.T1, signals={})

    registry = ModelRegistry(
        models=(ModelInfo(name="light", provider=Provider.OLLAMA, tier=Tier.T1),)
    )
    router = MultiModelRouter(
        registry,
        CaptureScorer(),  # type: ignore[arg-type]
        routing_context_chars=64,
    )
    await router.route(
        ChatRequest(
            model=None,
            messages=[
                ChatMessage(role="user", content="old " * 100),
                ChatMessage(role="user", content="latest task"),
            ],
        )
    )

    assert len(captured) == 1
    assert len(captured[0]) <= 64
    assert "latest task" in captured[0]


@pytest.mark.asyncio
async def test_router_falls_back_when_scorer_misses_deadline() -> None:
    started = Event()
    release = Event()

    class BlockingScorer:
        def score(self, _prompt: str) -> ScoringResult:
            started.set()
            release.wait(timeout=1)
            return ScoringResult(score=1.0, tier=Tier.T3, signals={})

    registry = ModelRegistry(
        models=(ModelInfo(name="light", provider=Provider.OLLAMA, tier=Tier.T1),)
    )
    router = MultiModelRouter(
        registry,
        BlockingScorer(),  # type: ignore[arg-type]
        scoring_timeout_ms=50,
    )
    decision = await router.route(
        ChatRequest(model=None, messages=[ChatMessage(role="user", content="say hello")])
    )
    release.set()

    assert started.is_set()
    assert decision.tier == Tier.T1


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


@pytest.mark.asyncio
async def test_zai_catalog_prefers_flash_for_simple_and_glm53_for_complex() -> None:
    from llmrouter.core.registry import load_model_registry

    registry = load_model_registry(
        "config/models.yaml",
        benchmark_catalog_path="data/model_benchmarks.yaml",
    )
    router = MultiModelRouter(
        registry,
        PromptScorer(),
        provider_cost_order=["zai", "ollama"],
        dynamic_benchmark_routing=False,
    )

    simple = await router.route(
        ChatRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Resuma esta frase em uma linha.")],
        )
    )
    complex_request = await router.route(
        ChatRequest(
            model=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Faça uma auditoria de segurança procurando vulnerabilidades "
                        "OWASP e injeção SQL."
                    ),
                )
            ],
        )
    )

    assert simple.primary.name == "zhipu/glm-5.3-flash"
    assert complex_request.primary.name == "zhipu/glm-5.3"
