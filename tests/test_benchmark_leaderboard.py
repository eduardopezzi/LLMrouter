from __future__ import annotations

from llmrouter.cli_panel import render_benchmark_leaderboards
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, Tier


def test_render_benchmark_leaderboards_orders_each_benchmark_independently() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="model-a",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=2,
                benchmark_scores=(("LiveCodeBench", 80.0), ("Codeforces", 2200.0)),
            ),
            ModelInfo(
                name="model-b",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=1,
                benchmark_scores=(("LiveCodeBench", 90.0), ("Codeforces", 1800.0)),
            ),
            ModelInfo(
                name="model-c",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=3,
                benchmark_scores=(("LiveCodeBench", 70.0),),
            ),
            ModelInfo(
                name="model-d",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                priority=4,
                benchmark_scores=(("LiveCodeBench", 99.0),),
            ),
        )
    )

    rendered = render_benchmark_leaderboards(registry)

    assert "Benchmark leaders (top 3 by normalized score)" in rendered
    assert "Coverage: 4/4 configured model(s)" in rendered
    assert "LiveCodeBench — 4 candidates, showing top 3" in rendered
    livecodebench = rendered.split("LiveCodeBench", 1)[1]
    assert livecodebench.index("model-d") < livecodebench.index("model-b")
    assert livecodebench.index("model-b") < livecodebench.index("model-a")
    codeforces = rendered.split("Codeforces", 1)[1].split("LiveCodeBench", 1)[0]
    assert codeforces.index("model-a") < codeforces.index("model-b")


def test_render_benchmark_leaderboards_explains_when_scores_are_missing() -> None:
    assert render_benchmark_leaderboards(ModelRegistry()) == (
        "Benchmark leaders: no published scores loaded."
    )


def test_render_benchmark_leaderboards_marks_single_candidate_as_insufficient() -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="only-model",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                benchmark_scores=(("PrivateBench", 75.0),),
            ),
        )
    )

    rendered = render_benchmark_leaderboards(registry)

    assert "Coverage: 1/1 configured model(s)" in rendered
    assert "PrivateBench — insufficient coverage: 1 candidate" in rendered
