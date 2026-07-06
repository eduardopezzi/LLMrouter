"""Tests for main.py entrypoint and CLI argument parsing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from llmrouter.main import (
    _parse_args,
    _build_health_tracker,
    _build_health_tracker_from_settings,
    main,
)


class TestArgParsing:
    """Test CLI argument parsing without running the server."""

    def test_parse_args_default(self) -> None:
        with patch.object(sys, "argv", ["llmrouter"]):
            args = _parse_args()
        assert args.command is None
        assert args.debug is False

    def test_parse_args_export_contracts(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "export-contracts", "--output", "out.json",
            "--service", "test-svc",
        ]):
            args = _parse_args()
        assert args.command == "export-contracts"
        assert args.output == "out.json"
        assert args.service == "test-svc"

    def test_parse_args_check_contracts(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "check-contracts", "prev.json", "curr.json",
        ]):
            args = _parse_args()
        assert args.command == "check-contracts"
        assert args.previous == "prev.json"
        assert args.current == "curr.json"

    def test_parse_args_diff_contracts(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "diff-contracts", "a.json", "b.json",
        ]):
            args = _parse_args()
        assert args.command == "diff-contracts"

    def test_parse_args_publish_contracts(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "publish-contracts", "--repo", "https://example.com/repo.git",
        ]):
            args = _parse_args()
        assert args.command == "publish-contracts"
        assert args.repo == "https://example.com/repo.git"

    def test_parse_args_panel_stats(self) -> None:
        with patch.object(sys, "argv", ["llmrouter", "panel", "--stats"]):
            args = _parse_args()
        assert args.command == "panel"
        assert args.stats is True

    def test_parse_args_panel_list_priorities(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "panel", "--list-model-priorities", "--priority-limit", "5",
        ]):
            args = _parse_args()
        assert args.command == "panel"
        assert args.list_model_priorities is True
        assert args.priority_limit == 5

    def test_parse_args_panel_promote_model(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "panel", "--promote-model", "gpt-4o",
        ]):
            args = _parse_args()
        assert args.command == "panel"
        assert args.promote_model == "gpt-4o"

    def test_parse_args_panel_set_strategy(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "panel", "--set-strategy", "quality",
        ]):
            args = _parse_args()
        assert args.command == "panel"
        assert args.set_strategy == "quality"

    def test_parse_args_panel_set_fallback_count(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "panel", "--set-fallback-count", "3",
        ]):
            args = _parse_args()
        assert args.command == "panel"
        assert args.set_fallback_count == 3

    def test_parse_args_panel_set_provider_cost_order(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "panel", "--set-provider-cost-order", "deepseek,zai,ollama",
        ]):
            args = _parse_args()
        assert args.command == "panel"
        assert args.set_provider_cost_order == "deepseek,zai,ollama"

    def test_parse_args_health(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "health", "--backend", "sqlite", "--db-path", "/tmp/health.db",
            "--window-minutes", "30", "--json",
        ]):
            args = _parse_args()
        assert args.command == "health"
        assert args.backend == "sqlite"
        assert args.db_path == "/tmp/health.db"
        assert args.window_minutes == 30
        assert args.json is True

    def test_parse_args_server_options(self) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "--host", "0.0.0.0", "--port", "8080", "--workers", "4", "--debug",
        ]):
            args = _parse_args()
        assert args.host == "0.0.0.0"
        assert args.port == 8080
        assert args.workers == 4
        assert args.debug is True


class TestHealthTrackerBuilders:
    """Test health tracker builder functions."""

    def test_build_health_tracker_from_settings(self) -> None:
        from llmrouter.config import Settings
        from llmrouter.core.health import HealthWeights, InMemoryHealthStore
        settings = Settings()
        tracker = _build_health_tracker_from_settings(settings)
        assert tracker.window_minutes == settings.health.window_minutes
        assert isinstance(tracker.store, InMemoryHealthStore)

    def test_build_health_tracker_memory(self) -> None:
        args = argparse.Namespace(backend="memory", db_path="data/health.db", window_minutes=15)
        tracker = _build_health_tracker(args)
        from llmrouter.core.health import InMemoryHealthStore
        assert isinstance(tracker.store, InMemoryHealthStore)
        assert tracker.window_minutes == 15

    def test_build_health_tracker_sqlite(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            backend="sqlite",
            db_path=str(tmp_path / "health.db"),
            window_minutes=30,
        )
        tracker = _build_health_tracker(args)
        from llmrouter.core.health import SQLiteHealthStore
        assert isinstance(tracker.store, SQLiteHealthStore)
        assert tracker.window_minutes == 30


class TestMainCommands:
    """Test main() subcommand dispatching."""

    def test_main_export_contracts(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        models_file = tmp_path / "models.yaml"
        models_file.write_text("models:\n  - name: test-model\n    provider: openai\n    tier: 1\n")
        output = tmp_path / "contract.json"

        with patch.object(sys, "argv", [
            "llmrouter", "export-contracts",
            "--models-file", str(models_file),
            "--output", str(output),
            "--service", "test",
        ]):
            main()

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["service"] == "test"
        captured = capsys.readouterr()
        assert "Exported" in captured.out

    def test_main_check_contracts_compatible(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        snap = {"schema_version": "1.0", "endpoints": [], "models": [], "routing_roles": []}
        f1 = tmp_path / "prev.json"
        f2 = tmp_path / "curr.json"
        f1.write_text(json.dumps(snap))
        f2.write_text(json.dumps(snap))

        with patch.object(sys, "argv", [
            "llmrouter", "check-contracts", str(f1), str(f2),
        ]):
            main()

        captured = capsys.readouterr()
        assert "No contract changes" in captured.out

    def test_main_check_contracts_breaking(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f1 = tmp_path / "prev.json"
        f2 = tmp_path / "curr.json"
        f1.write_text(json.dumps({"schema_version": "1.0"}))
        f2.write_text(json.dumps({"schema_version": "2.0"}))

        with patch.object(sys, "argv", [
            "llmrouter", "check-contracts", str(f1), str(f2),
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_diff_contracts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f1 = tmp_path / "prev.json"
        f2 = tmp_path / "curr.json"
        f1.write_text(json.dumps({"schema_version": "1.0"}))
        f2.write_text(json.dumps({"schema_version": "2.0"}))

        with patch.object(sys, "argv", [
            "llmrouter", "diff-contracts", str(f1), str(f2),
        ]):
            main()

        captured = capsys.readouterr()
        assert "breaking" in captured.out

    def test_main_panel_stats(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        models_file = tmp_path / "models.yaml"
        models_file.write_text("models:\n  - name: m1\n    provider: openai\n    tier: 1\n")

        with patch.object(sys, "argv", [
            "llmrouter", "panel",
            "--models-file", str(models_file),
            "--stats",
        ]):
            main()

        captured = capsys.readouterr()
        assert "LLMrouter CLI Panel" in captured.out

    def test_main_panel_list_priorities(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        models_file = tmp_path / "models.yaml"
        models_file.write_text("models:\n  - name: m1\n    provider: openai\n    tier: 1\n")

        with patch.object(sys, "argv", [
            "llmrouter", "panel",
            "--models-file", str(models_file),
            "--list-model-priorities",
        ]):
            main()

        captured = capsys.readouterr()
        assert "m1" in captured.out

    def test_main_panel_promote_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        models_file = tmp_path / "models.yaml"
        models_file.write_text(
            "models:\n  - name: m1\n    provider: openai\n    tier: 1\n"
            "  - name: m2\n    provider: ollama\n    tier: 2\n"
        )

        with patch.object(sys, "argv", [
            "llmrouter", "panel",
            "--models-file", str(models_file),
            "--promote-model", "m2",
        ]):
            main()

        captured = capsys.readouterr()
        assert "Promoted" in captured.out
        content = models_file.read_text()
        # m2 should now be priority 1
        assert "priority: 1" in content

    def test_main_panel_set_strategy(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("")

        with patch.object(sys, "argv", [
            "llmrouter", "panel",
            "--env-file", str(env_file),
            "--set-strategy", "quality",
        ]):
            main()

        captured = capsys.readouterr()
        assert "quality" in captured.out
        assert "LLMROUTER_ROUTING__STRATEGY=quality" in env_file.read_text()

    def test_main_panel_set_fallback_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("")

        with patch.object(sys, "argv", [
            "llmrouter", "panel",
            "--env-file", str(env_file),
            "--set-fallback-count", "5",
        ]):
            main()

        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "LLMROUTER_ROUTING__FALLBACK_COUNT=5" in env_file.read_text()

    def test_main_panel_set_provider_cost_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("")

        with patch.object(sys, "argv", [
            "llmrouter", "panel",
            "--env-file", str(env_file),
            "--set-provider-cost-order", "deepseek,zai,ollama",
        ]):
            main()

        captured = capsys.readouterr()
        assert "deepseek" in captured.out

    def test_main_health_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "health", "--backend", "memory", "--json",
        ]):
            main()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "window_minutes" in data
        assert "models" in data

    def test_main_health_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(sys, "argv", [
            "llmrouter", "health", "--backend", "memory",
        ]):
            main()

        captured = capsys.readouterr()
        assert "Model health" in captured.out or "no data" in captured.out