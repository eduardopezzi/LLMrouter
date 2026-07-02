from __future__ import annotations

import json
import os
import subprocess
import sys

from llmrouter.contract_publisher import github_token_from_env
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, Tier
from llmrouter.cross_repository import (
    BreakingChangeDetector,
    ChangeSeverity,
    ContractRegistry,
    resolve_project_contract_path,
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(
                name="ollama/test",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                capabilities=frozenset({"review", "fix"}),
                context_window=8192,
                max_tokens=8192,
                priority=1,
                api_base="http://localhost:11434",
            ),
        )
    )


def test_contract_registry_exports_deterministic_snapshot() -> None:
    snapshot = ContractRegistry(_registry()).snapshot()

    assert snapshot["schema_version"] == "1.0"
    assert snapshot["service"] == "llmrouter"
    assert snapshot["routing_roles"] == ["fix", "review"]

    models = snapshot["models"]
    assert isinstance(models, list)
    assert models[0] == {
        "id": "ollama/test",
        "provider": "ollama",
        "provider_model": "test",
        "tier": 2,
        "capabilities": ["fix", "review"],
        "context_window": 8192,
        "max_tokens": 8192,
        "priority": 1,
        "api_base": "http://localhost:11434",
    }

    endpoints = snapshot["endpoints"]
    assert isinstance(endpoints, list)
    assert "/v1/chat/completions" in {endpoint["path"] for endpoint in endpoints}


def test_breaking_change_detector_flags_removed_capability_and_smaller_context() -> None:
    previous = ContractRegistry(_registry()).snapshot()
    current_registry = ModelRegistry(
        models=(
            ModelInfo(
                name="ollama/test",
                provider=Provider.OLLAMA,
                tier=Tier.T2,
                capabilities=frozenset({"review"}),
                context_window=4096,
                max_tokens=4096,
                priority=1,
            ),
        )
    )
    current = ContractRegistry(current_registry).snapshot()

    result = BreakingChangeDetector().compare(previous, current)

    assert not result.is_compatible
    assert {change.path for change in result.breaking_changes} == {
        "models.ollama/test.capabilities",
        "models.ollama/test.context_window",
        "routing_roles",
    }
    assert all(change.severity == ChangeSeverity.BREAKING for change in result.breaking_changes)


def test_breaking_change_detector_allows_added_model_and_capability() -> None:
    previous = ContractRegistry(_registry()).snapshot()
    expanded = _registry().extend(
        (
            ModelInfo(
                name="ollama/new",
                provider=Provider.OLLAMA,
                tier=Tier.T1,
                capabilities=frozenset({"documentation"}),
            ),
        )
    )
    current = ContractRegistry(expanded).snapshot()

    result = BreakingChangeDetector().compare(previous, current)

    assert result.is_compatible
    assert {change.path for change in result.non_breaking_changes} == {
        "models.ollama/new",
        "routing_roles",
    }


def test_contract_cli_exports_and_checks_snapshots(tmp_path) -> None:
    snapshot = tmp_path / "llmrouter.contract.json"
    env = {**os.environ, "PYTHONPATH": "src"}
    export = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmrouter.main",
            "export-contracts",
            "--models-file",
            "config/models.example.yaml",
            "--output",
            str(snapshot),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert export.returncode == 0
    assert snapshot.exists()

    data = json.loads(snapshot.read_text(encoding="utf-8"))
    assert data["service"] == "llmrouter"
    assert len(data["models"]) == 22

    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmrouter.main",
            "check-contracts",
            str(snapshot),
            str(snapshot),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert check.returncode == 0
    assert "No contract changes detected." in check.stdout


def test_project_contract_path_resolves_project_case_insensitively(tmp_path) -> None:
    project_dir = tmp_path / "LLMRouter"
    project_dir.mkdir()

    path = resolve_project_contract_path(
        tmp_path,
        "llmrouter",
        "current.json",
        create=True,
    )

    assert path == project_dir / "current.json"


def test_contract_cli_exports_to_shared_repo_project_folder(tmp_path) -> None:
    project_dir = tmp_path / "LLMRouter"
    project_dir.mkdir()
    env = {**os.environ, "PYTHONPATH": "src"}

    export = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmrouter.main",
            "export-contracts",
            "--models-file",
            "config/models.example.yaml",
            "--contracts-root",
            str(tmp_path),
            "--project",
            "llmrouter",
            "--filename",
            "current.json",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert export.returncode == 0
    output = project_dir / "current.json"
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["service"] == "llmrouter"


def test_github_token_can_be_loaded_from_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=secret-token\n", encoding="utf-8")

    assert github_token_from_env(env_file) == "secret-token"
