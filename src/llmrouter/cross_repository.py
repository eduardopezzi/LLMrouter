"""Cross-repository contract snapshots and compatibility checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from llmrouter.api.routes import ChatCompletionPayload, SemanticInspectPayload
from llmrouter.core.registry import ModelRegistry

CONTRACT_SCHEMA_VERSION = "1.0"


class ChangeSeverity(str, Enum):
    """Compatibility severity for contract differences."""

    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"


@dataclass(frozen=True)
class ContractChange:
    """A single compatibility-relevant contract difference."""

    severity: ChangeSeverity
    path: str
    message: str
    before: object = None
    after: object = None


@dataclass(frozen=True)
class ContractCheckResult:
    """Result of comparing an old contract snapshot with a new one."""

    changes: tuple[ContractChange, ...] = ()

    @property
    def breaking_changes(self) -> tuple[ContractChange, ...]:
        return tuple(
            change for change in self.changes if change.severity == ChangeSeverity.BREAKING
        )

    @property
    def non_breaking_changes(self) -> tuple[ContractChange, ...]:
        return tuple(
            change for change in self.changes if change.severity == ChangeSeverity.NON_BREAKING
        )

    @property
    def is_compatible(self) -> bool:
        return not self.breaking_changes


@dataclass(frozen=True)
class ContractRegistry:
    """Build JSON contract snapshots consumed by peer repositories."""

    registry: ModelRegistry
    service_name: str = "llmrouter"
    schema_version: str = CONTRACT_SCHEMA_VERSION
    endpoints: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable contract snapshot."""
        endpoints = self.endpoints or _default_endpoints()
        return {
            "schema_version": self.schema_version,
            "service": self.service_name,
            "endpoints": sorted(endpoints, key=lambda endpoint: str(endpoint["path"])),
            "models": [
                {
                    "id": model.name,
                    "provider": model.provider.value,
                    "provider_model": model.provider_model_name,
                    "tier": model.tier.value,
                    "capabilities": sorted(model.capabilities),
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                    "priority": model.priority,
                    "api_base": model.api_base,
                }
                for model in sorted(self.registry.all(), key=lambda model: model.name)
            ],
            "routing_roles": sorted(
                {
                    capability
                    for model in self.registry.all()
                    for capability in model.capabilities
                }
            ),
        }

    def write_snapshot(self, path: str | Path) -> None:
        """Write the contract snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class BreakingChangeDetector:
    """Detect breaking and non-breaking differences between snapshots."""

    def compare(
        self,
        previous: dict[str, object],
        current: dict[str, object],
    ) -> ContractCheckResult:
        changes: list[ContractChange] = []
        self._compare_schema_version(previous, current, changes)
        self._compare_endpoints(previous, current, changes)
        self._compare_models(previous, current, changes)
        self._compare_routing_roles(previous, current, changes)
        return ContractCheckResult(tuple(changes))

    def compare_files(
        self,
        previous_path: str | Path,
        current_path: str | Path,
    ) -> ContractCheckResult:
        """Load and compare two JSON snapshots."""
        previous = load_contract_snapshot(previous_path)
        current = load_contract_snapshot(current_path)
        return self.compare(previous, current)

    def _compare_schema_version(
        self,
        previous: dict[str, object],
        current: dict[str, object],
        changes: list[ContractChange],
    ) -> None:
        before = previous.get("schema_version")
        after = current.get("schema_version")
        if before == after:
            return
        changes.append(
            ContractChange(
                severity=ChangeSeverity.BREAKING,
                path="schema_version",
                message="Contract schema version changed",
                before=before,
                after=after,
            )
        )

    def _compare_endpoints(
        self,
        previous: dict[str, object],
        current: dict[str, object],
        changes: list[ContractChange],
    ) -> None:
        before = _by_key(previous.get("endpoints"), "path")
        after = _by_key(current.get("endpoints"), "path")

        for path, endpoint in before.items():
            if path not in after:
                changes.append(
                    ContractChange(
                        severity=ChangeSeverity.BREAKING,
                        path=f"endpoints.{path}",
                        message="Endpoint was removed",
                        before=endpoint,
                    )
                )
                continue
            current_endpoint = after[path]
            for field_name in ("method", "request_schema"):
                if endpoint.get(field_name) != current_endpoint.get(field_name):
                    changes.append(
                        ContractChange(
                            severity=ChangeSeverity.BREAKING,
                            path=f"endpoints.{path}.{field_name}",
                            message=f"Endpoint {field_name} changed",
                            before=endpoint.get(field_name),
                            after=current_endpoint.get(field_name),
                        )
                    )

        for path, endpoint in after.items():
            if path not in before:
                changes.append(
                    ContractChange(
                        severity=ChangeSeverity.NON_BREAKING,
                        path=f"endpoints.{path}",
                        message="Endpoint was added",
                        after=endpoint,
                    )
                )

    def _compare_models(
        self,
        previous: dict[str, object],
        current: dict[str, object],
        changes: list[ContractChange],
    ) -> None:
        before = _by_key(previous.get("models"), "id")
        after = _by_key(current.get("models"), "id")

        for model_id, model in before.items():
            if model_id not in after:
                changes.append(
                    ContractChange(
                        severity=ChangeSeverity.BREAKING,
                        path=f"models.{model_id}",
                        message="Model was removed",
                        before=model,
                    )
                )
                continue
            current_model = after[model_id]
            self._compare_required_model_fields(model_id, model, current_model, changes)

        for model_id, model in after.items():
            if model_id not in before:
                changes.append(
                    ContractChange(
                        severity=ChangeSeverity.NON_BREAKING,
                        path=f"models.{model_id}",
                        message="Model was added",
                        after=model,
                    )
                )

    def _compare_required_model_fields(
        self,
        model_id: str,
        previous: dict[str, object],
        current: dict[str, object],
        changes: list[ContractChange],
    ) -> None:
        for field_name in ("provider", "provider_model", "tier"):
            if previous.get(field_name) != current.get(field_name):
                changes.append(
                    ContractChange(
                        severity=ChangeSeverity.BREAKING,
                        path=f"models.{model_id}.{field_name}",
                        message=f"Model {field_name} changed",
                        before=previous.get(field_name),
                        after=current.get(field_name),
                    )
                )

        previous_capabilities = set(_string_list(previous.get("capabilities")))
        current_capabilities = set(_string_list(current.get("capabilities")))
        removed = sorted(previous_capabilities - current_capabilities)
        added = sorted(current_capabilities - previous_capabilities)
        if removed:
            changes.append(
                ContractChange(
                    severity=ChangeSeverity.BREAKING,
                    path=f"models.{model_id}.capabilities",
                    message="Model capabilities were removed",
                    before=removed,
                    after=sorted(current_capabilities),
                )
            )
        if added:
            changes.append(
                ContractChange(
                    severity=ChangeSeverity.NON_BREAKING,
                    path=f"models.{model_id}.capabilities",
                    message="Model capabilities were added",
                    before=sorted(previous_capabilities),
                    after=added,
                )
            )

        previous_context = int(previous.get("context_window") or 0)
        current_context = int(current.get("context_window") or 0)
        if current_context < previous_context:
            changes.append(
                ContractChange(
                    severity=ChangeSeverity.BREAKING,
                    path=f"models.{model_id}.context_window",
                    message="Model context window decreased",
                    before=previous_context,
                    after=current_context,
                )
            )
        elif current_context > previous_context:
            changes.append(
                ContractChange(
                    severity=ChangeSeverity.NON_BREAKING,
                    path=f"models.{model_id}.context_window",
                    message="Model context window increased",
                    before=previous_context,
                    after=current_context,
                )
            )

    def _compare_routing_roles(
        self,
        previous: dict[str, object],
        current: dict[str, object],
        changes: list[ContractChange],
    ) -> None:
        before = set(_string_list(previous.get("routing_roles")))
        after = set(_string_list(current.get("routing_roles")))
        removed = sorted(before - after)
        added = sorted(after - before)
        if removed:
            changes.append(
                ContractChange(
                    severity=ChangeSeverity.BREAKING,
                    path="routing_roles",
                    message="Routing roles were removed",
                    before=removed,
                    after=sorted(after),
                )
            )
        if added:
            changes.append(
                ContractChange(
                    severity=ChangeSeverity.NON_BREAKING,
                    path="routing_roles",
                    message="Routing roles were added",
                    before=sorted(before),
                    after=added,
                )
            )


def load_contract_snapshot(path: str | Path) -> dict[str, object]:
    """Load a JSON contract snapshot."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("contract snapshot must be a JSON object")
    return data


def resolve_project_contract_path(
    root: str | Path,
    project: str,
    filename: str,
    *,
    create: bool = False,
) -> Path:
    """Resolve a project contract path under a shared contracts repository.

    Project directory lookup is case-insensitive so ``LLMrouter``, ``llmrouter``
    and ``LLMRouter`` all resolve to the same existing folder.
    """
    if not project:
        raise ValueError("project name is required")
    if Path(filename).name != filename:
        raise ValueError("contract filename must not contain directories")

    root_path = Path(root)
    project_dir = _find_project_dir(root_path, project)
    if project_dir is None:
        project_dir = root_path / project
        if not create:
            raise FileNotFoundError(f"project folder not found: {project}")
        project_dir.mkdir(parents=True, exist_ok=True)
    elif create:
        project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / filename


def format_contract_changes(result: ContractCheckResult) -> str:
    """Return a compact human-readable diff report."""
    if not result.changes:
        return "No contract changes detected."
    lines = []
    for change in result.changes:
        lines.append(f"{change.severity.value}: {change.path}: {change.message}")
    return "\n".join(lines)


def _default_endpoints() -> tuple[dict[str, object], ...]:
    return (
        {
            "path": "/health",
            "method": "GET",
            "auth_required": False,
            "response_schema": {
                "status": "str",
                "models": "int",
                "providers": "list[str]",
                "evaluator": "bool",
                "openai_compatible": "object",
            },
        },
        {
            "path": "/v1/models",
            "method": "GET",
            "auth_required": True,
            "response_schema": {"object": "str", "data": "list[model]"},
        },
        {
            "path": "/v1/chat/completions",
            "method": "POST",
            "auth_required": True,
            "request_schema": _model_schema(ChatCompletionPayload),
            "response_schema": {"object": "str", "choices": "list[choice]", "usage": "object"},
            "streaming": True,
        },
        {
            "path": "/v1/llmrouter/semantic/inspect",
            "method": "POST",
            "auth_required": True,
            "request_schema": _model_schema(SemanticInspectPayload),
            "response_schema": {
                "score": "float",
                "tier": "int",
                "semantic_role": "str",
                "semantic_confidence": "float",
                "semantic_used": "bool",
                "signals": "object",
            },
        },
        {
            "path": "/admin/evaluator/run-cycle",
            "method": "POST",
            "auth_required": True,
            "response_schema": {
                "evaluated": "int",
                "optimal": "int",
                "correct": "int",
                "overkill": "int",
                "underkill": "int",
            },
        },
    )


def _model_schema(model_type: type[BaseModel]) -> dict[str, object]:
    schema = model_type.model_json_schema()
    return {
        "required": schema.get("required", []),
        "properties": sorted((schema.get("properties") or {}).keys()),
    }


def _by_key(value: object, key: str) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_key = item.get(key)
        if isinstance(raw_key, str) and raw_key:
            result[raw_key] = item
    return result


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _find_project_dir(root: Path, project: str) -> Path | None:
    if not root.exists():
        return None
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    normalized = project.casefold()
    for child in root.iterdir():
        if child.is_dir() and child.name.casefold() == normalized:
            return child
    return None
