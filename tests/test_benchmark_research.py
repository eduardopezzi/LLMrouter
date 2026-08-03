from __future__ import annotations

import asyncio
import json

from llmrouter.benchmark_research import BenchmarkResearcher, BenchmarkResearchReport
from llmrouter.benchmark_scheduler import BenchmarkRefreshScheduler


class _FakeResearcher(BenchmarkResearcher):
    async def _ask_llm(self, prompt: str):  # type: ignore[no-untyped-def]
        assert "Source excerpts" in prompt
        return {
            "source_reviews": [
                {
                    "url": "https://example.test/card",
                    "assessment": "approved_candidate",
                    "reason": "Official model card with a table.",
                }
            ],
            "source_proposals": [
                {
                    "model": "ollama/candidate:cloud",
                    "url": "https://example.test/candidate",
                    "source_type": "official_model_card",
                    "reason": "Candidate source.",
                    "confidence": 0.8,
                }
            ],
            "model_proposals": [
                {
                    "name": "ollama/candidate:cloud",
                    "provider": "ollama",
                    "reason": "Published benchmarks found.",
                    "source_urls": ["https://example.test/candidate"],
                    "confidence": 0.8,
                }
            ],
        }


def test_researcher_writes_review_only_proposals(tmp_path) -> None:
    sources = tmp_path / "sources.yaml"
    catalog = tmp_path / "catalog.yaml"
    proposals = tmp_path / "proposals.json"
    sources.write_text(
        "sources:\n  - model: ollama/example:cloud\n    url: https://example.test/card\n"
    )
    catalog.write_text("schema_version: 1\nmodels: {}\n")
    researcher = _FakeResearcher(
        base_url="http://localhost:11434",
        model="research-model",
        proposal_path=str(proposals),
    )

    report = asyncio.run(
        researcher.research(
            sources,
            catalog,
            fetch_source=lambda _url: _async_value("<table>official benchmark evidence</table>"),
        )
    )

    saved = json.loads(proposals.read_text())
    assert report == BenchmarkResearchReport(1, 1, 1, proposals)
    assert saved["status"] == "pending_human_review"
    assert saved["source_proposals"][0]["model"] == "ollama/candidate:cloud"
    assert "non-authoritative" in saved["notice"]


def test_scheduler_runs_research_after_a_refresh(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    class FakeResearcher:
        async def research(self, sources_path: str, catalog_path: str) -> BenchmarkResearchReport:
            calls.append((sources_path, catalog_path))
            return BenchmarkResearchReport(0, 0, 0, tmp_path / "proposals.json")

    monkeypatch.setattr(
        "llmrouter.benchmark_scheduler.refresh_benchmark_catalog",
        lambda *_args, **_kwargs: _refresh_report(tmp_path),
    )
    scheduler = BenchmarkRefreshScheduler(
        sources_path="sources.yaml",
        catalog_path="catalog.yaml",
        interval_seconds=1,
        researcher=FakeResearcher(),  # type: ignore[arg-type]
    )

    asyncio.run(scheduler.refresh_once())

    assert calls == [("sources.yaml", "catalog.yaml")]


async def _async_value(value: str) -> str:
    return value


def _refresh_report(tmp_path):  # type: ignore[no-untyped-def]
    from llmrouter.benchmark_catalog import RefreshReport

    return RefreshReport(0, 0, False, tmp_path / "catalog.yaml")
