# ruff: noqa: UP017
"""LLM-assisted, review-only research for benchmark sources and model coverage."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx
import yaml


@dataclass(frozen=True)
class BenchmarkResearchReport:
    """Result of one proposal-only research cycle."""

    source_reviews: int
    source_proposals: int
    model_proposals: int
    output_path: Path


class BenchmarkResearcher:
    """Ask an LLM to assess sources and write proposals for human review.

    This class deliberately never edits ``benchmark_sources.yaml``, the model
    catalog, or benchmark scores. A browse-capable model may discover new URLs;
    models without browsing can still evaluate the source excerpts supplied by
    the deterministic refresh, but must not claim web verification.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        proposal_path: str,
        models_path: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        internet_search_enabled: bool = True,
        internet_search_max_results: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.proposal_path = Path(proposal_path)
        self.models_path = Path(models_path) if models_path else None
        self.api_key = api_key
        self.timeout = timeout
        self.internet_search_enabled = internet_search_enabled
        self.internet_search_max_results = internet_search_max_results

    async def research(
        self,
        sources_path: str | Path,
        catalog_path: str | Path,
        *,
        fetch_source: Callable[[str], Awaitable[str]] | None = None,
        search_web: Callable[[str], Awaitable[list[dict[str, str]]]] | None = None,
    ) -> BenchmarkResearchReport:
        """Evaluate declared sources and persist pending, non-authoritative proposals."""
        sources = _load_yaml(Path(sources_path))
        catalog = _load_yaml(Path(catalog_path))
        raw_sources = sources.get("sources", [])
        source_excerpts = await _source_excerpts(raw_sources, fetch_source or self._fetch_source)
        uncovered_models = _uncovered_models(self.models_path, catalog)
        web_evidence = await _internet_evidence(
            uncovered_models,
            search_web or self._search_web,
            enabled=self.internet_search_enabled,
        )
        response = await self._ask_llm(
            _research_prompt(
                source_definitions=raw_sources,
                source_excerpts=source_excerpts,
                catalog=catalog,
                uncovered_models=uncovered_models,
                web_evidence=web_evidence,
            )
        )
        proposal = _validated_proposal(response)
        proposal.update(
            {
                "schema_version": 1,
                "status": "pending_human_review",
                "generated_at": _timestamp(),
                "research_model": self.model,
                "source_excerpts_available": len(source_excerpts),
                "uncovered_models_checked": len(uncovered_models),
                "internet_search_enabled": self.internet_search_enabled,
                "notice": (
                    "Suggestions are non-authoritative. Review official URLs, methodology, "
                    "and table mappings before adding a source or model to active catalogs."
                ),
            }
        )
        self.proposal_path.parent.mkdir(parents=True, exist_ok=True)
        self.proposal_path.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return BenchmarkResearchReport(
            source_reviews=len(proposal["source_reviews"]),
            source_proposals=len(proposal["source_proposals"]),
            model_proposals=len(proposal["model_proposals"]),
            output_path=self.proposal_path,
        )

    async def _fetch_source(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            response = await client.get(
                url, headers={"User-Agent": "LLMrouter benchmark research/1.0"}
            )
            response.raise_for_status()
            return response.text

    async def _search_web(self, query: str) -> list[dict[str, str]]:
        """Return compact search evidence; the LLM still must verify identity.

        This is deliberately a discovery mechanism. Search hits are never
        converted into scores or active sources by this module.
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        document = await self._fetch_source(url)
        results: list[dict[str, str]] = []
        for href, title, snippet in re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
            document,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            clean_url = _search_result_url(href)
            if not clean_url:
                continue
            results.append(
                {
                    "url": clean_url,
                    "title": _plain_html(title)[:300],
                    "snippet": _plain_html(snippet)[:800],
                }
            )
            if len(results) >= self.internet_search_max_results:
                break
        return results

    async def _ask_llm(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious benchmark-research assistant. Return only JSON. "
                        "Never invent scores, URLs, methodology, browsing, or verification."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        ) as client:
            response = await client.post("/api/chat", json=payload, headers=headers)
            response.raise_for_status()
        return _extract_json(_ollama_content(response.json()))


async def _source_excerpts(
    raw_sources: object,
    fetch_source: Callable[[str], Awaitable[str]],
) -> list[dict[str, str]]:
    if not isinstance(raw_sources, list):
        return []
    excerpts: list[dict[str, str]] = []
    for source in raw_sources:
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            continue
        url = source["url"]
        try:
            document = await fetch_source(url)
        except (httpx.HTTPError, OSError, ValueError):
            continue
        plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", document)).strip()
        excerpts.append({"url": url, "content": plain[:8_000]})
    return excerpts


def _research_prompt(
    *,
    source_definitions: object,
    source_excerpts: list[dict[str, str]],
    catalog: dict[str, Any],
    uncovered_models: list[dict[str, str]],
    web_evidence: list[dict[str, Any]],
) -> str:
    return (
        """Review benchmark evidence for an LLM router.

The active score catalog is authoritative only after deterministic extraction
from a manually approved official source. Your output is a human-review queue,
not an update. Internet search results below are discovery evidence only. For
each uncovered model, assess whether a result identifies the exact model,
variant, size, and mode. Cloud-provider aliases must be searched using the
canonical name shown in the uncovered-model list: for example,
`ollama/deepseek-v4-pro:cloud` becomes `deepseek v4 pro`.

Never provide a benchmark score. Never treat a similar family, another size,
or another mode as an exact match. Propose an official model card or an
official benchmark page only when the evidence is sufficient; otherwise state
why the model remains unresolved. LangDB and other aggregators may be cited as
discovery evidence, but are not by themselves an approved score source.

Return one JSON object with exactly these list keys:
{
  "source_reviews": [{"url":"...","assessment":"...","reason":"..."}],
  "source_proposals": [{"model":"...","url":"https://...","reason":"..."}],
  "model_proposals": [{"name":"...","provider":"...","reason":"..."}]
}

Active source definitions:
"""
        + json.dumps(source_definitions, ensure_ascii=False)
        + "\n\nSource excerpts:\n"
        + json.dumps(source_excerpts, ensure_ascii=False)
        + "\n\nActive score catalog:\n"
        + json.dumps(catalog, ensure_ascii=False)
        + "\n\nUncovered configured models and canonical web queries:\n"
        + json.dumps(uncovered_models, ensure_ascii=False)
        + "\n\nInternet search evidence for uncovered models:\n"
        + json.dumps(web_evidence, ensure_ascii=False)
    )


async def _internet_evidence(
    uncovered_models: list[dict[str, str]],
    search_web: Callable[[str], Awaitable[list[dict[str, str]]]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    evidence: list[dict[str, Any]] = []
    for item in uncovered_models:
        query = f'{item["canonical_name"]} benchmark results'
        try:
            results = await search_web(query)
        except (httpx.HTTPError, OSError, ValueError):
            results = []
        evidence.append(
            {
                "model": item["model"],
                "canonical_name": item["canonical_name"],
                "query": query,
                "results": results[:10],
            }
        )
    return evidence


def _uncovered_models(models_path: Path | None, catalog: dict[str, Any]) -> list[dict[str, str]]:
    """Find configured models with no score record and create safe web queries."""
    if models_path is None:
        return []
    configured = _load_yaml(models_path).get("models", [])
    scored = catalog.get("models", {})
    if not isinstance(configured, list) or not isinstance(scored, dict):
        return []
    missing: list[dict[str, str]] = []
    for entry in configured:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name or _has_scores(scored.get(name)):
            continue
        missing.append({"model": name, "canonical_name": _canonical_model_name(name)})
    return missing


def _has_scores(entry: object) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("benchmark_scores"), dict) and bool(
        entry["benchmark_scores"]
    )


def _canonical_model_name(name: str) -> str:
    """Convert a provider alias into a human-searchable model name."""
    bare = name.split("/", 1)[-1].strip()
    bare = re.sub(r":(?:[^:]*-)?cloud$", "", bare, flags=re.IGNORECASE)
    bare = re.sub(r":", " ", bare)
    return re.sub(r"[-_]+", " ", bare).strip()


def _plain_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _search_result_url(value: str) -> str | None:
    """Unwrap DuckDuckGo redirect URLs and accept HTTPS evidence only."""
    match = re.search(r"uddg=([^&]+)", value)
    url = re.sub(r"\+", " ", match.group(1)) if match else value
    # DDG's redirect values are percent encoded; using it without decoding is
    # not useful, while non-HTTPS pages must never be proposed as sources.
    from urllib.parse import unquote

    url = unquote(url)
    return url if url.startswith("https://") else None


def _validated_proposal(value: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for key in ("source_reviews", "source_proposals", "model_proposals"):
        raw = value.get(key, [])
        result[key] = (
            [item for item in raw if isinstance(item, dict)][:50] if isinstance(raw, list) else []
        )
    return result


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    value = json.loads(stripped[start : end + 1] if start >= 0 and end >= start else stripped)
    if not isinstance(value, dict):
        raise ValueError("research LLM response must be a JSON object")
    return value


def _ollama_content(body: dict[str, Any]) -> str:
    message = body.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(body.get("response"), str):
        return body["response"]
    raise ValueError("research LLM response did not include message.content")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
