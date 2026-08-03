from __future__ import annotations

from pathlib import Path

import pytest

from llmrouter.core.benchmark_affinity import BenchmarkAffinityScorer
from llmrouter.core.registry import ModelRegistry, load_model_registry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import ScoringResult
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingStrategy,
    Tier,
)


class _KeywordEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "terminal" in lowered or "bash" in lowered:
                vectors.append([1.0, 0.0])
            elif "matemática" in lowered or "equação" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.1, 0.1])
        return vectors


def _write_knowledge_base(path: Path) -> None:
    path.write_text(
        "benchmarks = {\n"
        "    'TerminalBench 2.0': {\n"
        "        'description': 'terminal bash linux',\n"
        "        'difficulty': 'expert',\n"
        "        'keywords': ['terminal', 'bash'],\n"
        "        'typical_prompts': ['Configure o terminal Bash.'],\n"
        "    },\n"
        "    'MATH': {\n"
        "        'description': 'matemática equação',\n"
        "        'difficulty': 'hard',\n"
        "        'keywords': ['matemática', 'equação'],\n"
        "        'typical_prompts': ['Resolva a equação.'],\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )


def test_benchmark_affinities_are_normalized(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "benchmarks.py"
    _write_knowledge_base(knowledge_path)
    scorer = BenchmarkAffinityScorer(
        _KeywordEmbedder(),
        knowledge_base_path=knowledge_path,
        embedding_cache_path=tmp_path / "cache.json",
        similarity_threshold=0.30,
        top_k=2,
    )

    result = scorer.score("Use bash no terminal para configurar o serviço")

    assert result.signals["benchmark_top"] == "TerminalBench 2.0"
    assert result.signals["benchmark_used"] is True
    assert sum(result.signals["benchmark_affinities"].values()) == pytest.approx(1.0)
    assert result.tier == Tier.T3


def test_benchmark_affinity_cache_is_reused(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "benchmarks.py"
    cache_path = tmp_path / "cache.json"
    _write_knowledge_base(knowledge_path)
    scorer = BenchmarkAffinityScorer(
        _KeywordEmbedder(),
        knowledge_base_path=knowledge_path,
        embedding_cache_path=cache_path,
    )
    scorer.score("terminal bash")

    assert cache_path.exists()
    assert "fingerprint" in cache_path.read_text(encoding="utf-8")


class _FixedBenchmarkScorer:
    def score(self, prompt: str) -> ScoringResult:
        return ScoringResult(
            score=0.7,
            tier=Tier.T2,
            signals={
                "benchmark_top": "TerminalBench 2.0",
                "benchmark_affinities": {"TerminalBench 2.0": 1.0},
            },
        )


@pytest.mark.asyncio
async def test_router_uses_prompt_specific_benchmark_scores() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="weak-terminal",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=1,
                benchmark_scores=(("TerminalBench 2.0", 20.0),),
            ),
            ModelInfo(
                name="strong-terminal",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=20,
                benchmark_scores=(("TerminalBench 2.0", 90.0),),
            ),
        )
    )
    router = MultiModelRouter(
        registry,
        _FixedBenchmarkScorer(),
        strategy=RoutingStrategy.QUALITY,
    )

    decision = await router.route(
        ChatRequest(
            model=None,
            messages=[ChatMessage(role="user", content="Configure este serviço no terminal")],
        )
    )

    assert decision.primary.name == "strong-terminal"
    assert "benchmark=TerminalBench 2.0" in decision.reason


def test_registry_loads_raw_benchmark_scores(tmp_path: Path) -> None:
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        "models:\n"
        "  - name: test/model\n"
        "    provider: ollama\n"
        "    benchmark_scores:\n"
        "      MMLU-Pro: 84.2\n"
        "      Codeforces: 2816\n",
        encoding="utf-8",
    )

    model = load_model_registry(catalog).models[0]

    assert dict(model.benchmark_scores) == {"Codeforces": 2816.0, "MMLU-Pro": 84.2}
