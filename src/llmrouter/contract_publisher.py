"""Publish contract snapshots to a shared GitHub repository."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from llmrouter.core.registry import ModelRegistry
from llmrouter.cross_repository import ContractRegistry, resolve_project_contract_path


@dataclass(frozen=True)
class ContractPublishResult:
    """Result of publishing a contract snapshot."""

    changed: bool
    contract_path: str
    commit_sha: str | None = None


@dataclass(frozen=True)
class ContractPublisher:
    """Publish the current project contract into a central versions repository."""

    repository_url: str = "https://github.com/Vieli-Tech/phoenix_versions.git"
    branch: str = "main"
    project: str = "llmrouter"
    filename: str = "llmrouter.contract.json"
    service_name: str = "llmrouter"

    def publish(
        self,
        registry: ModelRegistry,
        *,
        github_token: str | None = None,
    ) -> ContractPublishResult:
        """Clone, update, commit and push a contract snapshot."""
        token = github_token or github_token_from_env()
        if not token:
            raise RuntimeError("GITHUB_TOKEN is required to publish contracts")

        with tempfile.TemporaryDirectory(prefix="llmrouter-contracts-") as workspace:
            repo_dir = Path(workspace) / "phoenix_versions"
            git_env = _git_auth_env(token)
            _run_git(
                [
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    self.branch,
                    self.repository_url,
                    str(repo_dir),
                ],
                env=git_env,
            )

            contract_path = resolve_project_contract_path(
                repo_dir,
                self.project,
                self.filename,
                create=True,
            )
            ContractRegistry(registry=registry, service_name=self.service_name).write_snapshot(
                contract_path
            )

            if not _has_changes(repo_dir):
                return ContractPublishResult(
                    changed=False,
                    contract_path=_relative_path(repo_dir, contract_path),
                )

            _run_git(["add", _relative_path(repo_dir, contract_path)], cwd=repo_dir, env=git_env)
            _run_git(
                [
                    "-c",
                    "user.name=llmrouter-contract-bot",
                    "-c",
                    "user.email=llmrouter-contract-bot@users.noreply.github.com",
                    "commit",
                    "-m",
                    f"Update {self.project} contract",
                ],
                cwd=repo_dir,
                env=git_env,
            )
            _run_git(["push", "origin", self.branch], cwd=repo_dir, env=git_env)
            commit_sha = _run_git(["rev-parse", "HEAD"], cwd=repo_dir, env=git_env).strip()
            return ContractPublishResult(
                changed=True,
                contract_path=_relative_path(repo_dir, contract_path),
                commit_sha=commit_sha,
            )


def github_token_from_env(env_file: str | Path = ".env") -> str | None:
    """Load GITHUB_TOKEN from the process environment or a local .env file."""
    load_dotenv(env_file, override=False)
    token = os.environ.get("GITHUB_TOKEN")
    return token.strip() if token else None


def _git_auth_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: bearer {token}"
    return env


def _has_changes(repo_dir: Path) -> bool:
    status = _run_git(["status", "--porcelain"], cwd=repo_dir)
    return bool(status.strip())


def _relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git executable was not found")
    result = subprocess.run(
        [git, *args],
        check=False,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(message)
    return result.stdout
