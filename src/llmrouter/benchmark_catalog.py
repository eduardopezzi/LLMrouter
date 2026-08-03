# ruff: noqa: UP017
"""Versioned benchmark-score catalog and refresh helpers.

Benchmark refreshes are deliberately kept outside the request path.  A source
definition identifies an official, structured table; this module extracts only
the declared cells and keeps provenance with every value.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml


class BenchmarkRefreshError(RuntimeError):
    """Raised when a configured benchmark source cannot be validated."""


@dataclass(frozen=True)
class RefreshReport:
    """Outcome of an attempted benchmark catalog refresh."""

    models_updated: int
    scores_updated: int
    changed: bool
    output_path: Path


class _HTMLTableParser(HTMLParser):
    """Small dependency-free HTML table parser for trusted model-card tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def refresh_benchmark_catalog(
    sources_path: str | Path,
    output_path: str | Path,
    *,
    fetch: Callable[[str, float], str] | None = None,
    timeout: float = 30.0,
    now: datetime | None = None,
    write: bool = True,
) -> RefreshReport:
    """Fetch configured official tables and update a provenance-rich YAML catalog.

    Existing values are retained until a source validates successfully.  A
    refresh is all-or-nothing: no output is written when any source fails.
    """
    source_file = Path(sources_path)
    destination = Path(output_path)
    sources = _load_sources(source_file)
    current = _load_yaml(destination) if destination.exists() else {}
    catalog: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": current.get("generated_at", _timestamp(now)),
        "models": dict(current.get("models", {}))
        if isinstance(current.get("models"), dict)
        else {},
    }
    downloader = fetch or _fetch_url
    updated_models = 0
    updated_scores = 0

    for source in sources:
        try:
            scores = _extract_source_scores(source, downloader, timeout)
        except Exception as exc:  # Preserve the previous approved catalog on any bad source.
            raise BenchmarkRefreshError(f"{source['model']}: {exc}") from exc
        model_entry = catalog["models"].setdefault(source["model"], {})
        if not isinstance(model_entry, dict):
            model_entry = {}
            catalog["models"][source["model"]] = model_entry
        benchmark_scores = model_entry.setdefault("benchmark_scores", {})
        if not isinstance(benchmark_scores, dict):
            benchmark_scores = {}
            model_entry["benchmark_scores"] = benchmark_scores
        updated_models += 1
        for benchmark, value in scores.items():
            candidate = {
                "value": value,
                "source_url": source["url"],
                "source_type": source["source_type"],
                "collected_at": _timestamp(now),
                "published_at": source.get("published_at"),
                "methodology": source.get("methodology", ""),
            }
            existing = benchmark_scores.get(benchmark)
            if not isinstance(existing, dict) or any(
                existing.get(key) != candidate[key]
                for key in ("value", "source_url", "source_type", "published_at", "methodology")
            ):
                benchmark_scores[benchmark] = candidate
                updated_scores += 1

    changed = _canonical_yaml(catalog) != _canonical_yaml(current)
    if changed:
        catalog["generated_at"] = _timestamp(now)
    if changed and write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_canonical_yaml(catalog), encoding="utf-8")
    return RefreshReport(updated_models, updated_scores, changed, destination)


def load_catalog_scores(path: str | Path) -> dict[str, dict[str, float]]:
    """Load valid raw score values from the generated benchmark catalog."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    models = _load_yaml(catalog_path).get("models", {})
    if not isinstance(models, dict):
        return {}
    result: dict[str, dict[str, float]] = {}
    for model, entry in models.items():
        if not isinstance(model, str) or not isinstance(entry, dict):
            continue
        raw_scores = entry.get("benchmark_scores", {})
        if not isinstance(raw_scores, dict):
            continue
        scores: dict[str, float] = {}
        for benchmark, details in raw_scores.items():
            if not isinstance(benchmark, str) or not isinstance(details, dict):
                continue
            value = details.get("value")
            if isinstance(value, bool):
                continue
            try:
                scores[benchmark] = float(value)
            except (TypeError, ValueError):
                continue
        if scores:
            result[model] = scores
    return result


def _load_sources(path: Path) -> list[dict[str, Any]]:
    raw = _load_yaml(path).get("sources", [])
    if not isinstance(raw, list) or not raw:
        raise BenchmarkRefreshError("sources file must contain a non-empty 'sources' list")
    sources: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise BenchmarkRefreshError("each source must be a mapping")
        model = entry.get("model")
        url = entry.get("url")
        source_type = entry.get("source_type")
        metrics = entry.get("metrics")
        if not isinstance(model, str) or not model:
            raise BenchmarkRefreshError("source missing model")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise BenchmarkRefreshError(f"{model}: URL must use HTTPS")
        if source_type not in {"official", "official_model_card", "benchmark_leaderboard"}:
            raise BenchmarkRefreshError(f"{model}: unsupported source_type")
        if not isinstance(metrics, dict) or not metrics:
            raise BenchmarkRefreshError(f"{model}: metrics must be a non-empty mapping")
        if entry.get("format", "html_table") != "html_table":
            raise BenchmarkRefreshError(f"{model}: only html_table sources are supported")
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metrics.items()
        ):
            raise BenchmarkRefreshError(f"{model}: metrics must map names to table row labels")
        sources.append(dict(entry))
    return sources


def _extract_source_scores(
    source: dict[str, Any],
    fetch: Callable[[str, float], str],
    timeout: float,
) -> dict[str, float]:
    document = fetch(source["url"], timeout)
    parser = _HTMLTableParser()
    parser.feed(document)
    benchmark_column = str(source.get("benchmark_column", "Benchmark"))
    model_column = str(source.get("model_column", ""))
    if not model_column:
        raise BenchmarkRefreshError("model_column is required")
    table = _find_table(parser.tables, benchmark_column, model_column)
    headers = table[0]
    benchmark_index = _find_column(headers, benchmark_column)
    model_index = _find_column(headers, model_column)
    rows = {
        _normalized(row[benchmark_index]): row
        for row in table[1:]
        if len(row) > max(benchmark_index, model_index)
    }
    extracted: dict[str, float] = {}
    for benchmark, source_label in source["metrics"].items():
        row = rows.get(_normalized(source_label))
        if row is None:
            raise BenchmarkRefreshError(f"missing declared benchmark row '{source_label}'")
        extracted[benchmark] = _parse_score(row[model_index])
    return extracted


def _find_table(
    tables: list[list[list[str]]], benchmark_column: str, model_column: str
) -> list[list[str]]:
    for table in tables:
        if (
            table
            and _find_column(table[0], benchmark_column, required=False) is not None
            and _find_column(table[0], model_column, required=False) is not None
        ):
            return table
    raise BenchmarkRefreshError("declared table and columns were not found")


def _find_column(headers: list[str], name: str, *, required: bool = True) -> int | None:
    wanted = _normalized(name)
    for index, header in enumerate(headers):
        if _normalized(header) == wanted:
            return index
    if required:
        raise BenchmarkRefreshError(f"missing table column '{name}'")
    return None


def _parse_score(value: str) -> float:
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*%?\s*", value)
    if not match:
        raise BenchmarkRefreshError(f"invalid benchmark score '{value}'")
    score = float(match.group(1))
    if score < 0:
        raise BenchmarkRefreshError(f"negative benchmark score '{value}'")
    return score


def _fetch_url(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "LLMrouter benchmark refresh/1.0"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is validated from trusted config.
        return response.read().decode("utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise BenchmarkRefreshError(f"{path}: expected a YAML mapping")
    return data


def _canonical_yaml(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=True, default_flow_style=False)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _timestamp(now: datetime | None) -> str:
    instant = now or datetime.now(timezone.utc)
    return (
        instant.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
