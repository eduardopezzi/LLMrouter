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
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.proposal_path = Path(proposal_path)
        self.api_key = api_key
        self.timeout = timeout

    async def research(
        self,
        sources_path: str | Path,
        catalog_path: str | Path,
        *,
        fetch_source: Callable[[str], Awaitable[str]] | None = None,
    ) -> BenchmarkResearchReport:
        """Evaluate declared sources and persist pending, non-authoritative proposals."""
        sources = _load_yaml(Path(sources_path))
        catalog = _load_yaml(Path(catalog_path))
        raw_sources = sources.get("sources", [])
        source_excerpts = await _source_excerpts(raw_sources, fetch_source or self._fetch_source)
        response = await self._ask_llm(
            _research_prompt(
                source_definitions=raw_sources,
                source_excerpts=source_excerpts,
                catalog=catalog,
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
) -> str:
    return (
        """Review benchmark evidence for an LLM router.

The active score catalog is authoritative only after deterministic extraction
from a manually approved official source. Your output is a human-review queue,
not an update. If you have web-search tools, you may discover candidates; if
you do not, only assess the provided excerpts and leave unverified proposals
empty. Never provide a benchmark score.

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
    )


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
