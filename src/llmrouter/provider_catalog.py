"""Provider model discovery, catalog reconciliation, and change reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace
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
    complete_inventory: bool = False
    model_allowlist: frozenset[str] | None = None
    model_denylist: frozenset[str] = frozenset()


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
    reactivated_models: tuple[dict[str, str], ...] = ()


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
    apply_catalog: bool = False,
    write: bool = True,
) -> ProviderSyncReport:
    """Check official sources and optionally reconcile the active model catalog.

    New identifiers can be added from official inventories or documentation.
    Existing models are retired only after two successful checks against a
    source explicitly marked as a complete inventory. A failed source never
    removes models.
    """
    sources = _load_sources(Path(sources_path))
    previous = _load_snapshot(Path(snapshot_path))
    registry = load_model_registry(models_path, benchmark_catalog_path=benchmark_catalog_path)
    models = registry.all()
    configured_ids_by_provider = _configured_model_ids(Path(models_path))
    active_by_provider: dict[str, list[Any]] = {}
    for model in models:
        active_by_provider.setdefault(model.provider.value, []).append(model)
    active_providers = {model.provider.value for model in models} | set(configured_ids_by_provider)
    sources = [source for source in sources if source.provider in active_providers]

    source_records: dict[str, dict[str, Any]] = {}
    source_errors: list[dict[str, str]] = []
    discovered: dict[str, dict[str, dict[str, Any]]] = {}
    inventory_discovered: dict[str, dict[str, dict[str, Any]]] = {}
    changed_source_urls: set[str] = set()
    failed_inventory_providers: set[str] = set()
    model_allowlists_by_provider: dict[str, set[str]] = {}
    model_denylists_by_provider: dict[str, set[str]] = {}
    for source in sources:
        if source.model_allowlist is not None:
            model_allowlists_by_provider.setdefault(source.provider, set()).update(
                source.model_allowlist
            )
        model_denylists_by_provider.setdefault(source.provider, set()).update(
            source.model_denylist
        )

    for source in sources:
        try:
            body, model_records = _fetch_source(source, timeout=timeout)
            model_records = [
                record
                for record in model_records
                if isinstance(record.get("id"), str)
                and (
                    source.model_allowlist is None
                    or record["id"] in source.model_allowlist
                )
                and record["id"] not in source.model_denylist
            ]
            if (
                (source.complete_inventory or source.kind == "model_api")
                and not model_records
                and configured_ids_by_provider.get(source.provider)
            ):
                raise ValueError("complete provider inventory returned no model identifiers")
        except (httpx.HTTPError, OSError, ValueError) as exc:
            if source.complete_inventory or source.kind == "model_api":
                failed_inventory_providers.add(source.provider)
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
            "complete_inventory": source.complete_inventory or source.kind == "model_api",
            "sha256": digest,
            "models": model_records,
        }
        if source_records[source.url]["complete_inventory"]:
            inventory_discovered.setdefault(source.provider, {})
        for record in model_records:
            model_id = record.get("id")
            if isinstance(model_id, str) and model_id:
                discovered.setdefault(source.provider, {}).setdefault(model_id, record)
                if source_records[source.url]["complete_inventory"]:
                    inventory_discovered[source.provider].setdefault(model_id, record)

    _populate_missing_model_counts(
        active_by_provider,
        previous,
        source_records,
        blocked_providers=failed_inventory_providers,
    )

    new_models, removed_models, updated_models = _model_diffs(
        active_by_provider,
        discovered,
        previous,
        source_records,
        configured_ids_by_provider=configured_ids_by_provider,
        blocked_inventory_providers=failed_inventory_providers,
        model_allowlists_by_provider=model_allowlists_by_provider,
        model_denylists_by_provider=model_denylists_by_provider,
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

    snapshot_records = dict(previous)
    snapshot_records.update(source_records)
    for url, record in list(snapshot_records.items()):
        if record.get("provider") in failed_inventory_providers and _is_complete_inventory(record):
            reset_record = dict(record)
            reset_record["missing_models"] = {}
            snapshot_records[url] = reset_record

    snapshot_changed = snapshot_records != previous
    if write and apply_catalog:
        added, reactivated = _apply_provider_catalog_changes(
            Path(models_path),
            removed_models,
            inventory_discovered,
        )
        changed = changed or bool(added or reactivated)
        report = replace(
            report,
            changed=changed,
            new_models=tuple(added),
            reactivated_models=tuple(reactivated),
        )
    if write and (changed or snapshot_changed or source_errors):
        _write_snapshot(Path(snapshot_path), snapshot_records)
        _write_report(report)
    if write and apply_priority and priority_changes:
        complete_order = list(priority_order)
        if apply_catalog:
            complete_order.extend(
                item["model"] for item in (*report.new_models, *report.reactivated_models)
                if item["model"] not in complete_order
            )
        set_model_priority_order(models_path, complete_order)
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
                complete_inventory=bool(item.get("complete_inventory", False)),
                model_allowlist=_source_model_ids(item, "model_allowlist", provider),
                model_denylist=(
                    _source_model_ids(item, "model_denylist", provider) or frozenset()
                ),
            )
        )
    return sources


def _source_model_ids(
    item: dict[str, Any],
    field: str,
    provider: str,
) -> frozenset[str] | None:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(model_id, str) for model_id in value):
        raise ValueError(f"{field} for {provider} must be a list of model IDs")
    model_ids = frozenset(model_id.strip() for model_id in value if model_id.strip())
    if field == "model_allowlist" and not model_ids:
        raise ValueError(f"{field} for {provider} must not be empty")
    return model_ids


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
    if "models" in payload:
        raw_models = payload["models"]
    elif "data" in payload:
        raw_models = payload["data"]
    else:
        raise ValueError(f"{provider} model API response has no model list")
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
    *,
    configured_ids_by_provider: dict[str, set[str]] | None = None,
    blocked_inventory_providers: set[str] | None = None,
    model_allowlists_by_provider: dict[str, set[str]] | None = None,
    model_denylists_by_provider: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    new_models: list[dict[str, str]] = []
    removed_models: list[dict[str, str]] = []
    updated_models: list[dict[str, str]] = []
    previous_models: dict[tuple[object, object], dict[str, Any]] = {}
    for source_record in previous.values():
        provider_name = source_record.get("provider")
        for model_record in source_record.get("models", []):
            if isinstance(model_record, dict):
                previous_models[(provider_name, model_record.get("id"))] = model_record
    allowlists = model_allowlists_by_provider or {}
    denylists = model_denylists_by_provider or {}
    for provider in set(allowlists) | set(denylists):
        allowed_ids = allowlists.get(provider)
        denied_ids = denylists.get(provider, set())
        for model in active_by_provider.get(provider, []):
            model_id = _model_id(provider, model.name)
            if model_id in denied_ids or (
                allowed_ids is not None and model_id not in allowed_ids
            ):
                removed_models.append(
                    {
                        "provider": provider,
                        "model": model.name,
                        "evidence": "excluded by explicit provider model allowlist/denylist",
                    }
                )
    current_inventory_records: dict[str, list[dict[str, Any]]] = {}
    for source_record in source_records.values():
        provider = source_record.get("provider")
        if (
            not isinstance(provider, str)
            or not _is_complete_inventory(source_record)
            or provider in (blocked_inventory_providers or set())
        ):
            continue
        current_inventory_records.setdefault(provider, []).append(source_record)

    for provider, records in discovered.items():
        active_ids = {
            _model_id(provider, model.name) for model in active_by_provider.get(provider, [])
        }
        known_ids = (configured_ids_by_provider or {}).get(provider, active_ids)
        for model_id, record in records.items():
            if model_id not in known_ids and (provider, model_id) not in previous_models:
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
        inventories = current_inventory_records.get(provider, [])
        if inventories:
            available_ids = {
                model_record.get("id")
                for inventory in inventories
                for model_record in inventory.get("models", [])
                if isinstance(model_record, dict)
            }
            for model in active_by_provider.get(provider, []):
                model_id = _model_id(provider, model.name)
                miss_count = max(
                    (
                        _safe_int(inventory.get("missing_models", {}).get(model_id, 0))
                        for inventory in inventories
                        if isinstance(inventory.get("missing_models", {}), dict)
                    ),
                    default=0,
                )
                if model_id not in available_ids and miss_count >= 2:
                    removed_models.append(
                        {
                            "provider": provider,
                            "model": model.name,
                            "evidence": "absent from two successful official inventory checks",
                        }
                    )
    return (
        sorted(new_models, key=lambda item: (item["provider"], item["model"])),
        sorted(
            {item["model"]: item for item in removed_models}.values(),
            key=lambda item: (item["provider"], item["model"]),
        ),
        sorted(updated_models, key=lambda item: (item["provider"], item["model"])),
    )


def _is_complete_inventory(record: dict[str, Any]) -> bool:
    """Whether a source claims to list every model available to its API."""
    return bool(record.get("complete_inventory", record.get("kind") == "model_api"))


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _configured_model_ids(path: Path) -> dict[str, set[str]]:
    """Read all configured model names, including entries currently disabled."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    model_rows = raw.get("models", []) if isinstance(raw, dict) else []
    configured: dict[str, set[str]] = {}
    if not isinstance(model_rows, list):
        return configured
    for row in model_rows:
        if not isinstance(row, dict):
            continue
        provider, name = row.get("provider"), row.get("name")
        if isinstance(provider, str) and isinstance(name, str):
            configured.setdefault(provider, set()).add(_model_id(provider, name))
    return configured


def _populate_missing_model_counts(
    active_by_provider: dict[str, list[Any]],
    previous: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    *,
    blocked_providers: set[str] | None = None,
) -> None:
    """Advance absence counts only for providers with a successful full inventory."""
    inventory_records: dict[str, list[dict[str, Any]]] = {}
    for record in source_records.values():
        provider = record.get("provider")
        if (
            isinstance(provider, str)
            and _is_complete_inventory(record)
            and provider not in (blocked_providers or set())
        ):
            inventory_records.setdefault(provider, []).append(record)

    previous_misses: dict[str, dict[str, int]] = {}
    for record in previous.values():
        provider = record.get("provider")
        missing = record.get("missing_models", {})
        if (
            not isinstance(provider, str)
            or not _is_complete_inventory(record)
            or not isinstance(missing, dict)
        ):
            continue
        provider_misses = previous_misses.setdefault(provider, {})
        for model_id, count in missing.items():
            provider_misses[str(model_id)] = max(
                provider_misses.get(str(model_id), 0), _safe_int(count)
            )

    for provider, records in inventory_records.items():
        available_ids = {
            model.get("id")
            for record in records
            for model in record.get("models", [])
            if isinstance(model, dict)
        }
        missing: dict[str, int] = {}
        for model in active_by_provider.get(provider, []):
            model_id = _model_id(provider, model.name)
            if model_id not in available_ids:
                missing[model_id] = previous_misses.get(provider, {}).get(model_id, 0) + 1
        for record in records:
            record["missing_models"] = dict(missing)


def _apply_provider_catalog_changes(
    path: Path,
    removed_models: list[dict[str, str]],
    discovered: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Append new models and enable/disable entries while preserving YAML comments."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    model_rows = raw.get("models", []) if isinstance(raw, dict) else []
    if not isinstance(model_rows, list):
        raise ValueError(f"models file must contain a top-level 'models' list: {path}")
    rows_by_name = {
        row.get("name"): row
        for row in model_rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    updates: dict[str, bool] = {}
    reactivated: list[dict[str, str]] = []
    added: list[dict[str, str]] = []

    for item in removed_models:
        name = item["model"]
        row = rows_by_name.get(name)
        if row is not None and row.get("enabled") is not False:
            updates[name] = False

    for provider, records in discovered.items():
        for model_id in records:
            name = _provider_model_name(provider, model_id)
            row = rows_by_name.get(name)
            if row is not None:
                if row.get("enabled") is False:
                    updates[name] = True
                    reactivated.append(
                        {
                            "provider": provider,
                            "model": name,
                            "evidence": "official provider source lists the model again",
                        }
                    )
                continue
            entry = _new_model_entry(provider, name, model_rows)
            rows_by_name[name] = entry
            model_rows.append(entry)
            added.append(
                {
                    "provider": provider,
                    "model": name,
                    "evidence": "official provider source",
                }
            )

    if not updates and not added:
        return added, reactivated

    text = path.read_text(encoding="utf-8") if path.exists() else "models:\n"
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    name_pattern = re.compile(r"^(\s*)-\s+name:\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = name_pattern.match(line)
        if match:
            starts.append((index, _unquote_scalar(match.group(2))))
    for position, (start, name) in enumerate(starts):
        if name not in updates:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        property_indent = name_pattern.match(lines[start]).group(1) + "  "  # type: ignore[union-attr]
        enabled_pattern = re.compile(r"^(\s*)enabled:\s*(?:true|false)\s*$", re.IGNORECASE)
        enabled_index = next(
            (i for i in range(start + 1, end) if enabled_pattern.match(lines[i])),
            None,
        )
        state_line = f"{property_indent}enabled: {'true' if updates[name] else 'false'}"
        if enabled_index is None:
            lines.insert(start + 1, state_line)
            starts = [
                (line_index + 1 if line_index > start else line_index, model_name)
                for line_index, model_name in starts
            ]
        else:
            lines[enabled_index] = state_line

    if added:
        if not any(re.match(r"^\s*models\s*:", line) for line in lines):
            lines.insert(0, "models:")
        if any(re.match(r"^\s*models\s*:\s*\[\s*\]\s*$", line) for line in lines):
            lines = [re.sub(r"^(\s*models\s*):\s*\[\s*\]\s*$", r"\1:", line) for line in lines]
        entries = [
            next(row for row in model_rows if row.get("name") == item["model"])
            for item in added
        ]
        serialized = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True)
        serialized_lines = [
            "  " + line if line else line
            for line in serialized.rstrip().splitlines()
        ]

        models_header = next(
            (i for i, line in enumerate(lines) if re.match(r"^models\s*:", line)),
            None,
        )
        if models_header is None:
            raise ValueError(f"could not locate the top-level models list in {path}")
        top_level_key = re.compile(r"^(?:[A-Za-z_][\w-]*|['\"][^'\"]+['\"])\s*:")
        section_end = next(
            (
                i
                for i in range(models_header + 1, len(lines))
                if top_level_key.match(lines[i])
            ),
            len(lines),
        )
        if section_end > 0 and lines[section_end - 1].strip():
            serialized_lines.insert(0, "")
        if section_end < len(lines) and lines[section_end].strip():
            serialized_lines.append("")
        lines[section_end:section_end] = serialized_lines

    rendered = "\n".join(lines) + "\n"
    try:
        validated = yaml.safe_load(rendered) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"provider catalog update would create invalid YAML: {path}") from exc
    if not isinstance(validated, dict) or not isinstance(validated.get("models", []), list):
        raise ValueError(f"provider catalog update produced an invalid models list: {path}")
    expected_names = {item["model"] for item in added}
    written_names = {
        row.get("name")
        for row in validated["models"]
        if isinstance(row, dict)
    }
    if not expected_names.issubset(written_names):
        raise ValueError(f"provider catalog update lost new model entries: {path}")

    path.write_text(rendered, encoding="utf-8")
    return added, reactivated


def _unquote_scalar(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _new_model_entry(provider: str, name: str, rows: list[Any]) -> dict[str, Any]:
    priorities = [
        _safe_int(row.get("priority", 0))
        for row in rows
        if isinstance(row, dict) and row.get("enabled") is not False
    ]
    lowered = name.lower()
    if any(marker in lowered for marker in ("pro", "max", "reason", "flagship", "ultra")):
        tier = 3
    elif any(marker in lowered for marker in ("flash", "mini", "nano", "small", "-3b", "-7b")):
        tier = 1
    else:
        tier = 2
    return {
        "name": name,
        "provider": provider,
        "enabled": True,
        # Provider discovery is evidence that an identifier exists, not that
        # it is ready for production traffic. Let an operator promote it via
        # the per-model rollout controls after reviewing its capabilities.
        "rollout_percentage": 0,
        "tier": tier,
        "priority": max(priorities, default=0) + 1,
        "roles": ["review", "documentation", "summarization"],
        "max_tokens": 8192,
        "context_window": 8192,
        "description": (
            "Automatically discovered from an official provider source. "
            "Review its pricing, context limit, and supported capabilities."
        ),
    }


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
