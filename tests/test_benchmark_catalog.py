# ruff: noqa: UP017
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmrouter.benchmark_catalog import (
    BenchmarkRefreshError,
    load_catalog_scores,
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


def test_registry_merges_refreshed_scores_over_manual_values(tmp_path) -> None:
    models = tmp_path / "models.yaml"
    catalog = tmp_path / "catalog.yaml"
    models.write_text(
        "models:\n  - name: ollama/example:cloud\n    provider: ollama\n"
        "    tier: 2\n    benchmark_scores:\n      LiveCodeBench: 10\n"
    )
    catalog.write_text(
        "schema_version: 1\nmodels:\n  ollama/example:cloud:\n    benchmark_scores:\n"
        "      LiveCodeBench:\n        value: 71.5\n"
    )

    registry = load_model_registry(models, benchmark_catalog_path=catalog)

    assert dict(registry.models[0].benchmark_scores) == {"LiveCodeBench": 71.5}
