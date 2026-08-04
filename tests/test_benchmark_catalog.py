# ruff: noqa: UP017
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmrouter.benchmark_catalog import (
    BenchmarkRefreshError,
    load_catalog_scores,
    load_catalog_source_urls,
    refresh_benchmark_catalog,
)
from llmrouter.core.registry import load_model_registry

SOURCE_YAML = """
sources:
  - model: "ollama/example:cloud"
    source_type: official_model_card
    url: "https://example.test/model-card"
    format: html_table
    benchmark_column: Benchmark
    model_column: Example Model
    methodology: Official configuration
    metrics:
      "LiveCodeBench": "LiveCodeBench v6"
      "TerminalBench 2.0": "TerminalBench 2.0"
"""

TABLE = """
<table>
  <tr><th>Benchmark</th><th>Example Model</th></tr>
  <tr><td>LiveCodeBench v6</td><td>71.5</td></tr>
  <tr><td>TerminalBench 2.0</td><td>42.0%</td></tr>
</table>
"""

MARKDOWN_SOURCE_YAML = """
sources:
  - model: "ollama/model-b:cloud"
    source_type: benchmark_leaderboard
    url: "https://example.test/model-card.md"
    format: markdown_table
    benchmark_column: Benchmark
    model_column: Model B
    metrics:
      "GPQA Diamond": "GPQA-Diamond"
      "SWE-Bench Pro": "SWE-bench Pro"
"""

MARKDOWN_TABLE = """
|Benchmark|Model A|Model B|
|:---|:---:|:---:|
|Reasoning|||
|GPQA-Diamond|88.0|91.2|
|SWE-bench Pro|61.0|62.1|
"""


def test_refresh_writes_provenance_and_is_idempotent(tmp_path) -> None:
    sources = tmp_path / "sources.yaml"
    output = tmp_path / "catalog.yaml"
    sources.write_text(SOURCE_YAML)
    frozen_now = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)

    report = refresh_benchmark_catalog(
        sources,
        output,
        fetch=lambda _url, _timeout: TABLE,
        now=frozen_now,
    )

    assert report.changed is True
    assert report.scores_updated == 2
    raw = load_catalog_scores(output)
    assert raw["ollama/example:cloud"] == {"LiveCodeBench": 71.5, "TerminalBench 2.0": 42.0}
    text = output.read_text()
    assert "official_model_card" in text
    assert "2026-08-03T10:00:00Z" in text
    assert load_catalog_source_urls(output) == {
        "ollama/example:cloud": ("https://example.test/model-card",)
    }

    second = refresh_benchmark_catalog(
        sources,
        output,
        fetch=lambda _url, _timeout: TABLE,
        now=datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc),
        write=False,
    )
    assert second.changed is False
    assert second.scores_updated == 0


def test_failed_refresh_does_not_overwrite_approved_catalog(tmp_path) -> None:
    sources = tmp_path / "sources.yaml"
    output = tmp_path / "catalog.yaml"
    sources.write_text(SOURCE_YAML)
    output.write_text("schema_version: 1\ngenerated_at: old\nmodels: {}\n")

    with pytest.raises(BenchmarkRefreshError):
        refresh_benchmark_catalog(sources, output, fetch=lambda _url, _timeout: "<table></table>")

    assert output.read_text() == "schema_version: 1\ngenerated_at: old\nmodels: {}\n"


def test_refresh_extracts_markdown_model_card_table(tmp_path) -> None:
    sources = tmp_path / "sources.yaml"
    output = tmp_path / "catalog.yaml"
    sources.write_text(MARKDOWN_SOURCE_YAML)

    report = refresh_benchmark_catalog(
        sources,
        output,
        fetch=lambda _url, _timeout: MARKDOWN_TABLE,
    )

    assert report.scores_updated == 2
    assert load_catalog_scores(output)["ollama/model-b:cloud"] == {
        "GPQA Diamond": 91.2,
        "SWE-Bench Pro": 62.1,
    }


def test_registry_merges_refreshed_scores_over_manual_values(tmp_path) -> None:
    models = tmp_path / "models.yaml"
    catalog = tmp_path / "catalog.yaml"
    models.write_text(
        "models:\n  - name: ollama/example:cloud\n    provider: ollama\n"
        "    tier: 2\n    benchmark_scores:\n      LiveCodeBench: 10\n"
        "    benchmark_sources:\n      - https://example.test/approved\n"
    )
    catalog.write_text(
        "schema_version: 1\nmodels:\n  ollama/example:cloud:\n    benchmark_scores:\n"
        "      LiveCodeBench:\n        value: 71.5\n"
        "        source_url: https://example.test/validated\n"
    )

    registry = load_model_registry(models, benchmark_catalog_path=catalog)

    assert dict(registry.models[0].benchmark_scores) == {"LiveCodeBench": 71.5}
    assert registry.models[0].benchmark_sources == (
        "https://example.test/approved",
        "https://example.test/validated",
    )
