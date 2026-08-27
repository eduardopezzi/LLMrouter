"""Safe discovery and review queue for model-catalog lifecycle changes.

Discovery is deterministic.  An LLM may enrich a proposal later, but it never
becomes evidence that a model exists or has been retired.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from llmrouter.core.cooldown import is_ollama_cloud_model
from llmrouter.core.types import ModelInfo, Provider


@dataclass(frozen=True)
class CatalogProposal:
    """One non-authoritative change awaiting human approval."""

    action: str  # add | verify_absent
    provider: str
    model: str
    reason: str
    evidence: str
    suggested_rollout_percentage: float = 0.0


@dataclass(frozen=True)
class CatalogSyncReport:
    """Result of reconciling provider inventory against the configured catalog."""

    configured_models: int
    discovered_models: int
    proposals: tuple[CatalogProposal, ...]
    output_path: Path


def reconcile_ollama_local_models(
    configured: list[ModelInfo], inventory: set[str]
) -> tuple[CatalogProposal, ...]:
    """Diff local Ollama tags without treating cloud aliases as local failures."""
    local = [
        model
        for model in configured
        if model.provider == Provider.OLLAMA and not is_ollama_cloud_model(model)
    ]
    configured_names = {model.provider_model_name for model in local}
    proposals: list[CatalogProposal] = []
    for name in sorted(inventory - configured_names):
        proposals.append(
            CatalogProposal(
                action="add",
                provider=Provider.OLLAMA.value,
                model=f"ollama/{name}",
                reason="Installed local Ollama model is absent from the configured catalog.",
                evidence="Ollama /api/tags inventory",
            )
        )
    for model in sorted(local, key=lambda item: item.name):
        if model.provider_model_name not in inventory:
            proposals.append(
                CatalogProposal(
                    action="verify_absent",
                    provider=Provider.OLLAMA.value,
                    model=model.name,
                    reason="Configured local model was not returned by Ollama.",
                    evidence="Ollama /api/tags inventory; cloud aliases are excluded",
                )
            )
    return tuple(proposals)


def fetch_ollama_local_inventory(base_url: str, *, timeout: float = 15.0) -> set[str]:
    """Read locally installed tags from Ollama; no model generation is performed."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    response = httpx.get(f"{root}/api/tags", timeout=timeout)
    response.raise_for_status()
    body: Any = response.json()
    models = body.get("models", []) if isinstance(body, dict) else []
    return {
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str) and model["name"]
    }


def write_catalog_proposals(
    output_path: str | Path,
    *,
    configured_models: int,
    inventory: set[str],
    proposals: tuple[CatalogProposal, ...],
) -> CatalogSyncReport:
    """Persist a review queue atomically; it never changes ``models.yaml``."""
    path = Path(output_path)
    payload = {
        "schema_version": 1,
        "status": "pending_human_review",
        "generated_at": datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 support.
        .replace(microsecond=0)
        .isoformat(),
        "configured_models": configured_models,
        "discovered_models": len(inventory),
        "proposals": [asdict(proposal) for proposal in proposals],
        "notice": (
            "Inventory is deterministic, but proposals never modify the active catalog. "
            "Review provider availability, roles, costs and capability before applying."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return CatalogSyncReport(configured_models, len(inventory), proposals, path)


def configured_models(path: str | Path) -> list[ModelInfo]:
    """Load the active model YAML without benchmark overlays."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    from llmrouter.core.registry import load_model_registry

    return load_model_registry(path).all() if isinstance(raw, dict) else []
