"""Weekly provider documentation and model-catalog monitoring.

The provider APIs and documentation are evidence only.  This module can
reorder the already approved catalog, but it never adds or removes an active
model automatically.  New model identifiers are emitted as review proposals.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from llmrouter.cli_panel import set_model_priority_order
from llmrouter.core.benchmark_scorer import rank_models
from llmrouter.core.registry import load_model_registry


@dataclass(frozen=True)
class ProviderSource:
    """One official documentation page or model inventory endpoint."""

    provider: str
    url: str
    kind: str = "documentation"  # documentation | model_api
    api_key_env: str | None = None


@dataclass(frozen=True)
class ProviderSyncReport:
    """Summary written by a provider synchronization cycle."""

    changed: bool
    generated_at: str
    sources_checked: int
    sources_changed: int
    changed_sources: tuple[str, ...]
    source_errors: tuple[dict[str, str], ...]
    new_models: tuple[dict[str, str], ...]
    removed_models: tuple[dict[str, str], ...]
    updated_models: tuple[dict[str, str], ...]
    priority_changes: tuple[dict[str, object], ...]
    priority_order: tuple[str, ...]
    report_path: Path


def refresh_provider_catalog(
    sources_path: str | Path,
    models_path: str | Path,
    snapshot_path: str | Path,
    report_path: str | Path,
    *,
    strategy: str,
    provider_cost_order: list[str],
    benchmark_catalog_path: str | Path = "data/model_benchmarks.yaml",
    timeout: float = 30.0,
    apply_priority: bool = False,
    write: bool = True,
) -> ProviderSyncReport:
    """Check official sources and optionally apply a validated priority order.

    A failed source is retained in the report and never causes configured
    models to be removed.  The priority order is calculated from the active
    local benchmark catalog, so a provider API outage cannot invent ranking
    data.
    """
    sources = _load_sources(Path(sources_path))
    previous = _load_snapshot(Path(snapshot_path))
    registry = load_model_registry(models_path, benchmark_catalog_path=benchmark_catalog_path)
    models = registry.all()
    active_by_provider: dict[str, list[Any]] = {}
    for model in models:
        active_by_provider.setdefault(model.provider.value, []).append(model)
    active_providers = {model.provider.value for model in models}
    sources = [source for source in sources if source.provider in active_providers]

    source_records: dict[str, dict[str, Any]] = {}
    source_errors: list[dict[str, str]] = []
    discovered: dict[str, dict[str, dict[str, Any]]] = {}
    changed_source_urls: set[str] = set()

    for source in sources:
        try:
            body, model_records = _fetch_source(source, timeout=timeout)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            source_errors.append(
                {"provider": source.provider, "url": source.url, "error": str(exc)[:500]}
            )
            continue
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        previous_record = previous.get(source.url, {})
        if previous_record.get("sha256") != digest:
            changed_source_urls.add(source.url)
        source_records[source.url] = {
            "provider": source.provider,
            "kind": source.kind,
            "sha256": digest,
            "models": model_records,
        }
        for record in model_records:
            model_id = record.get("id")
            if isinstance(model_id, str) and model_id:
                discovered.setdefault(source.provider, {}).setdefault(model_id, record)

    new_models, removed_models, updated_models = _model_diffs(
        active_by_provider,
        discovered,
        previous,
        source_records,
    )
    ordered_models = rank_models(
        models,
        strategy=strategy,
        provider_cost_order=provider_cost_order,
    )
    priority_order = tuple(model.name for model in ordered_models)
    current_order = tuple(
        model.name for model in sorted(models, key=lambda item: (item.priority, item.name))
    )
    priority_changes = tuple(
        {
            "model": name,
            "old_priority": current_order.index(name) + 1,
            "new_priority": priority_order.index(name) + 1,
        }
        for name in current_order
        if current_order.index(name) != priority_order.index(name)
    )

    changed = bool(
        changed_source_urls or new_models or removed_models or updated_models or priority_changes
    )
    generated_at = _timestamp()
    report = ProviderSyncReport(
        changed=changed,
        generated_at=generated_at,
        sources_checked=len(source_records),
        sources_changed=len(changed_source_urls),
        changed_sources=tuple(sorted(changed_source_urls)),
        source_errors=tuple(source_errors),
        new_models=tuple(new_models),
        removed_models=tuple(removed_models),
        updated_models=tuple(updated_models),
        priority_changes=priority_changes,
        priority_order=priority_order,
        report_path=Path(report_path),
    )

    if write and changed:
        snapshot_records = dict(previous)
        snapshot_records.update(source_records)
        _write_snapshot(Path(snapshot_path), snapshot_records)
        _write_report(report)
    if apply_priority and priority_changes:
        set_model_priority_order(models_path, list(priority_order))
    return report


def _load_sources(path: Path) -> list[ProviderSource]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError(f"provider sources file must contain a 'sources' list: {path}")
    sources: list[ProviderSource] = []
    for item in raw["sources"]:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        url = item.get("url")
        if not isinstance(provider, str) or not isinstance(url, str):
            continue
        kind = item.get("kind", "documentation")
        if kind not in {"documentation", "model_api"}:
            raise ValueError(f"unsupported source kind: {kind}")
        api_key_env = item.get("api_key_env")
        sources.append(
            ProviderSource(
                provider=provider,
                url=url,
                kind=kind,
                api_key_env=api_key_env if isinstance(api_key_env, str) else None,
            )
        )
    return sources


def _load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources", {}) if isinstance(raw, dict) else {}
    return sources if isinstance(sources, dict) else {}


def _fetch_source(source: ProviderSource, *, timeout: float) -> tuple[str, list[dict[str, Any]]]:
    headers = {"User-Agent": "LLMrouter provider catalog monitor/1.0"}
    api_key = os.environ.get(source.api_key_env) if source.api_key_env else None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.get(source.url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    body = response.text
    if source.kind == "model_api":
        return body, _parse_model_api(source.provider, response.json())
    return body, _extract_documentation_models(source.provider, body)


def _parse_model_api(provider: str, payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError(f"{provider} model API returned a non-object response")
    raw_models = payload.get("models", payload.get("data", []))
    if not isinstance(raw_models, list):
        raise ValueError(f"{provider} model API response has no model list")
    records: list[dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("name", item.get("id"))
        if isinstance(model_id, str) and model_id:
            records.append({"id": model_id, **{str(k): v for k, v in item.items() if k != "name"}})
    return records


def _extract_documentation_models(provider: str, body: str) -> list[dict[str, Any]]:
    """Extract conservative model-id candidates from official documentation."""
    plain = re.sub(r"<[^>]+>", " ", body).lower()
    if provider == "deepseek":
        pattern = r"\bdeepseek-[a-z0-9]+(?:-[a-z0-9]+)+\b"
    elif provider == "zai":
        pattern = r"\bglm-[a-z0-9]+(?:-[a-z0-9]+)*\b"
    else:
        return []
    return [{"id": value} for value in sorted(set(re.findall(pattern, plain)))]


def _provider_model_name(provider: str, model_id: str) -> str:
    if provider == "ollama":
        return f"ollama/{model_id}"
    if provider == "deepseek":
        return f"deepseek/{model_id}"
    if provider == "zai":
        return f"zhipu/{model_id}"
    return f"{provider}/{model_id}"


def _model_id(provider: str, model_name: str) -> str:
    prefix = {"ollama": "ollama/", "deepseek": "deepseek/", "zai": "zhipu/"}.get(
        provider, f"{provider}/"
    )
    return model_name.removeprefix(prefix)


def _model_diffs(
    active_by_provider: dict[str, list[Any]],
    discovered: dict[str, dict[str, dict[str, Any]]],
    previous: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    new_models: list[dict[str, str]] = []
    removed_models: list[dict[str, str]] = []
    updated_models: list[dict[str, str]] = []
    successful_api_providers = {
        record["provider"]
        for record in source_records.values()
        if record["kind"] == "model_api"
    }
    previous_models: dict[tuple[object, object], dict[str, Any]] = {}
    for source_record in previous.values():
        provider_name = source_record.get("provider")
        for model_record in source_record.get("models", []):
            if isinstance(model_record, dict):
                previous_models[(provider_name, model_record.get("id"))] = model_record
    previous_api_ids: dict[str, set[object]] = {}
    for source_record in previous.values():
        provider = source_record.get("provider")
        if source_record.get("kind") != "model_api" or not isinstance(provider, str):
            continue
        previous_api_ids.setdefault(provider, set()).update(
            model.get("id")
            for model in source_record.get("models", [])
            if isinstance(model, dict)
        )
    current_api_ids: dict[str, set[object]] = {}
    for source_record in source_records.values():
        provider = source_record.get("provider")
        if source_record.get("kind") != "model_api" or not isinstance(provider, str):
            continue
        current_api_ids.setdefault(provider, set()).update(
            model.get("id")
            for model in source_record.get("models", [])
            if isinstance(model, dict)
        )

    for provider, records in discovered.items():
        active_ids = {
            _model_id(provider, model.name) for model in active_by_provider.get(provider, [])
        }
        for model_id, record in records.items():
            if model_id not in active_ids and (provider, model_id) not in previous_models:
                new_models.append(
                    {
                        "provider": provider,
                        "model": _provider_model_name(provider, model_id),
                        "evidence": "official model inventory/documentation",
                    }
                )
            previous_record = previous_models.get((provider, model_id))
            if previous_record is not None and previous_record != record:
                updated_models.append(
                    {
                        "provider": provider,
                        "model": _provider_model_name(provider, model_id),
                        "evidence": "official model inventory metadata changed",
                    }
                )
        if provider in successful_api_providers:
            for model in active_by_provider.get(provider, []):
                model_id = _model_id(provider, model.name)
                if (
                    model_id in previous_api_ids.get(provider, set())
                    and model_id not in current_api_ids.get(provider, set())
                ):
                    removed_models.append(
                        {
                            "provider": provider,
                            "model": model.name,
                            "evidence": "official model inventory no longer lists it",
                        }
                    )
    return (
        sorted(new_models, key=lambda item: (item["provider"], item["model"])),
        sorted(removed_models, key=lambda item: (item["provider"], item["model"])),
        sorted(updated_models, key=lambda item: (item["provider"], item["model"])),
    )


def _write_snapshot(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "sources": records}
    path.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8")


def _write_report(report: ProviderSyncReport) -> None:
    report.report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["report_path"] = str(report.report_path)
    report.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017
