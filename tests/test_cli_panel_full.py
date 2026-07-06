"""Tests for CLI panel utility functions and rendering logic."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from llmrouter.cli_panel import (
    RoutingPanelConfig,
    ModelPriority,
    routing_panel_config,
    set_routing_strategy,
    set_fallback_count,
    set_provider_cost_order,
    model_priorities,
    render_model_priorities,
    promote_model_priority,
    set_model_priority_order,
    reset_model_priorities_to_catalog_order,
    demote_model_priority,
    update_env_file,
    catalog_stats,
    render_panel_summary,
    render_current_settings,
    observation_stats,
    _build_llm_priority_prompt,
    _response_text,
    _parse_llm_priority_order,
    _extract_json_object,
    _validate_model_order,
    _parse_provider_selection,
    _normalize_provider_order,
    _model_blocks,
    _strip_yaml_scalar,
    _line_indent,
    _table_exists,
    _format_mapping,
    _read_log_tail,
    follow_log_file,
    _journalctl_follow_command,
    _journalctl_available,
)
from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, Tier


def _models_file(tmp_path: Path, models: list[dict[str, Any]] | None = None) -> Path:
    if models is None:
        models = [
            {"name": "model-a", "provider": "openai", "tier": 1, "priority": 1},
            {"name": "model-b", "provider": "ollama", "tier": 2, "priority": 2},
            {"name": "model-c", "provider": "zai", "tier": 3, "priority": 3},
        ]
    path = tmp_path / "models.yaml"
    entries = []
    for m in models:
        entries.append(f"  - name: {m['name']}")
        entries.append(f"    provider: {m['provider']}")
        entries.append(f"    tier: {m['tier']}")
        if "priority" in m:
            entries.append(f"    priority: {m['priority']}")
    path.write_text("models:\n" + "\n".join(entries) + "\n")
    return path


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(name="m1", provider=Provider.OPENAI, tier=Tier.T1, priority=1, capabilities=frozenset({"code"})),
            ModelInfo(name="m2", provider=Provider.OLLAMA, tier=Tier.T2, priority=2, capabilities=frozenset({"review"})),
        )
    )


# ---------------------------------------------------------------------------
# Settings and display functions
# ---------------------------------------------------------------------------


def test_routing_panel_config() -> None:
    settings = Settings()
    config = routing_panel_config(settings)
    assert isinstance(config, RoutingPanelConfig)
    assert config.strategy == settings.routing.strategy.value


def test_set_routing_strategy(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("")
    set_routing_strategy(env_file, "quality")
    content = env_file.read_text()
    assert "LLMROUTER_ROUTING__STRATEGY=quality" in content


def test_set_routing_strategy_invalid() -> None:
    with pytest.raises(Exception):
        set_routing_strategy(".env", "invalid_strategy")


def test_set_fallback_count(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("")
    set_fallback_count(env_file, 5)
    assert "LLMROUTER_ROUTING__FALLBACK_COUNT=5" in env_file.read_text()


def test_set_fallback_count_negative() -> None:
    with pytest.raises(ValueError):
        set_fallback_count(".env", -1)


def test_set_provider_cost_order(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("")
    set_provider_cost_order(env_file, ["deepseek", "ollama"])
    content = env_file.read_text()
    assert "deepseek" in content


def test_model_priorities() -> None:
    reg = _registry()
    priorities = model_priorities(reg, limit=10)
    assert len(priorities) == 2
    assert priorities[0].name == "m1"
    assert priorities[0].rank == 1


def test_model_priorities_empty() -> None:
    reg = ModelRegistry()
    priorities = model_priorities(reg)
    assert len(priorities) == 0


def test_model_priorities_limit() -> None:
    reg = _registry()
    priorities = model_priorities(reg, limit=1)
    assert len(priorities) == 1


def test_render_model_priorities() -> None:
    reg = _registry()
    result = render_model_priorities(reg, limit=10)
    assert "Top" in result
    assert "m1" in result


def test_render_model_priorities_empty() -> None:
    reg = ModelRegistry()
    result = render_model_priorities(reg)
    assert "(catalog is empty)" in result


# ---------------------------------------------------------------------------
# Priority operations on YAML files
# ---------------------------------------------------------------------------


def test_promote_model_priority(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    promote_model_priority(models_file, "model-c")
    blocks = _model_blocks(models_file)
    priorities = {b.name: b.priority for b in blocks}
    assert priorities["model-c"] == 1


def test_promote_model_priority_not_found(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    with pytest.raises(ValueError, match="model not found"):
        promote_model_priority(models_file, "nonexistent")


def test_promote_model_priority_no_models(tmp_path: Path) -> None:
    models_file = tmp_path / "models.yaml"
    models_file.write_text("models: []\n")
    with pytest.raises(ValueError, match="does not contain"):
        promote_model_priority(models_file, "any")


def test_set_model_priority_order(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    set_model_priority_order(models_file, ["model-c", "model-b", "model-a"])
    blocks = _model_blocks(models_file)
    priorities = {b.name: b.priority for b in blocks}
    assert priorities["model-c"] == 1
    assert priorities["model-b"] == 2
    assert priorities["model-a"] == 3


def test_set_model_priority_order_wrong_count(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    with pytest.raises(ValueError):
        set_model_priority_order(models_file, ["model-a"])


def test_set_model_priority_order_unknown_model(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    with pytest.raises(ValueError):
        set_model_priority_order(models_file, ["a", "b", "c"])


def test_reset_model_priorities(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    # First change priorities
    set_model_priority_order(models_file, ["model-c", "model-a", "model-b"])
    # Then reset
    reset_model_priorities_to_catalog_order(models_file)
    blocks = _model_blocks(models_file)
    priorities = {b.name: b.priority for b in blocks}
    # Should follow original YAML order
    assert priorities["model-a"] == 1
    assert priorities["model-b"] == 2
    assert priorities["model-c"] == 3


def test_demote_model_priority(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    result = demote_model_priority(models_file, "model-a")
    assert result is True
    blocks = _model_blocks(models_file)
    priorities = {b.name: b.priority for b in blocks}
    assert priorities["model-a"] == 3


def test_demote_model_priority_already_last(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    result = demote_model_priority(models_file, "model-c")
    assert result is False


def test_demote_model_priority_not_found(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    with pytest.raises(ValueError):
        demote_model_priority(models_file, "nonexistent")


# ---------------------------------------------------------------------------
# Env file operations
# ---------------------------------------------------------------------------


def test_update_env_file_existing_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEY1=value1\nKEY2=value2\n")
    update_env_file(env_file, {"KEY1": "updated"})
    content = env_file.read_text()
    assert "KEY1=updated" in content
    assert "KEY2=value2" in content


def test_update_env_file_new_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEY1=value1\n")
    update_env_file(env_file, {"NEW_KEY": "new_value"})
    content = env_file.read_text()
    assert "NEW_KEY=new_value" in content


def test_update_env_file_nonexistent(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    update_env_file(env_file, {"KEY": "value"})
    assert "KEY=value" in env_file.read_text()


def test_update_env_file_preserves_comments(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nKEY1=value1\n")
    update_env_file(env_file, {"KEY1": "updated"})
    content = env_file.read_text()
    assert "# comment" in content


# ---------------------------------------------------------------------------
# Stats and rendering
# ---------------------------------------------------------------------------


def test_catalog_stats() -> None:
    reg = _registry()
    stats = catalog_stats(reg)
    assert stats["models"] == 2
    assert "openai" in stats["providers"]


def test_catalog_stats_empty() -> None:
    stats = catalog_stats(ModelRegistry())
    assert stats["models"] == 0


def test_observation_stats_nonexistent(tmp_path: Path) -> None:
    result = observation_stats(tmp_path / "nonexistent.db")
    assert result["observations"] == 0


def test_observation_stats_empty_db(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    db.write_text("")  # invalid db, should handle
    result = observation_stats(db)
    assert result["observations"] == 0


def test_observation_stats_with_data(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY, prompt TEXT, chosen_model TEXT,
                response TEXT, latency_ms REAL, cost_usd REAL,
                prompt_tokens INTEGER, completion_tokens INTEGER,
                scorer_score REAL, scorer_tier INTEGER,
                metadata_json TEXT, created_at TEXT
            );
            CREATE TABLE reviews (
                id INTEGER PRIMARY KEY, observation_id INTEGER,
                relevance INTEGER, accuracy INTEGER, completeness INTEGER,
                concision INTEGER, safety INTEGER, quality_overall REAL,
                grade TEXT, suggested_model TEXT, rationale TEXT,
                created_at TEXT
            );
        """)
        conn.execute(
            "INSERT INTO observations (prompt, chosen_model, response, latency_ms, cost_usd, "
            "prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("prompt", "model-a", "response", 100.0, 0.01, 10, 5)
        )
        conn.commit()
    result = observation_stats(db)
    assert result["observations"] == 1
    assert result["reviews"] == 0
    assert result["avg_latency_ms"] == 100.0


def test_render_panel_summary() -> None:
    settings = Settings()
    reg = _registry()
    result = render_panel_summary(settings, reg)
    assert "LLMrouter CLI Panel" in result


def test_render_current_settings() -> None:
    settings = Settings()
    reg = _registry()
    result = render_current_settings(settings, reg)
    assert "Current Settings" in result
    assert "strategy" in result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_build_llm_priority_prompt() -> None:
    models = list(_registry().all())
    prompt = _build_llm_priority_prompt(models, strategy="cost", provider_cost_order=["zai", "ollama"])
    assert "cost" in prompt
    assert "json" in prompt.lower()


def test_build_llm_priority_prompt_unknown_strategy() -> None:
    models = list(_registry().all())
    prompt = _build_llm_priority_prompt(models, strategy="unknown", provider_cost_order=[])
    assert "Rank models" in prompt


def test_response_text_from_message() -> None:
    choices = [{"message": {"content": "hello"}}]
    assert _response_text(choices) == "hello"


def test_response_text_from_text_field() -> None:
    choices = [{"text": "fallback"}]
    assert _response_text(choices) == "fallback"


def test_response_text_empty_choices() -> None:
    with pytest.raises(ValueError):
        _response_text([])


def test_response_text_not_list() -> None:
    with pytest.raises(ValueError):
        _response_text("not a list")


def test_response_text_no_content() -> None:
    with pytest.raises(ValueError):
        _response_text([{}])


def test_parse_llm_priority_order_valid() -> None:
    content = '{"models": ["a", "b", "c"]}'
    result = _parse_llm_priority_order(content)
    assert result == ["a", "b", "c"]


def test_parse_llm_priority_order_fenced() -> None:
    content = '```json\n{"models": ["a", "b"]}\n```'
    result = _parse_llm_priority_order(content)
    assert result == ["a", "b"]


def test_parse_llm_priority_order_with_text() -> None:
    content = 'Here is the result: {"models": ["x", "y"]}'
    result = _parse_llm_priority_order(content)
    assert result == ["x", "y"]


def test_parse_llm_priority_order_invalid_json() -> None:
    with pytest.raises(ValueError):
        _parse_llm_priority_order("not json at all")


def test_parse_llm_priority_order_not_list() -> None:
    content = '{"models": 123}'
    with pytest.raises(ValueError):
        _parse_llm_priority_order(content)


def test_parse_llm_priority_order_bare_list() -> None:
    content = '["a", "b"]'
    result = _parse_llm_priority_order(content)
    assert result == ["a", "b"]


def test_extract_json_object_fenced() -> None:
    result = _extract_json_object('```json\n{"a": 1}\n```')
    assert result == '{"a": 1}'


def test_extract_json_object_plain() -> None:
    result = _extract_json_object('prefix {"a": 1} suffix')
    assert result == '{"a": 1}'


def test_extract_json_object_not_found() -> None:
    assert _extract_json_object("no json here") is None


def test_validate_model_order_valid() -> None:
    _validate_model_order(["a", "b"], ["a", "b"])


def test_validate_model_order_wrong_count() -> None:
    with pytest.raises(ValueError):
        _validate_model_order(["a"], ["a", "b"])


def test_validate_model_order_missing() -> None:
    with pytest.raises(ValueError, match="missing"):
        _validate_model_order(["a", "c"], ["a", "b"])


def test_validate_model_order_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _validate_model_order(["a", "c"], ["a", "b"])


def test_validate_model_order_duplicate() -> None:
    with pytest.raises(ValueError):
        _validate_model_order(["a", "a"], ["a", "b"])


def test_parse_provider_selection_numeric() -> None:
    result = _parse_provider_selection("1,3", ["ollama", "openai", "zai"])
    assert result == ["ollama", "zai"]


def test_parse_provider_selection_numeric_out_of_range() -> None:
    result = _parse_provider_selection("5", ["ollama", "openai"])
    assert result == []


def test_parse_provider_selection_names() -> None:
    result = _parse_provider_selection("ollama,openai", ["ollama", "openai"])
    assert result == ["ollama", "openai"]


def test_parse_provider_selection_empty() -> None:
    assert _parse_provider_selection("", ["a"]) == []


def test_normalize_provider_order() -> None:
    result = _normalize_provider_order(["ollama", "zai"])
    assert result == ["ollama", "zai"]


def test_normalize_provider_order_empty() -> None:
    with pytest.raises(ValueError):
        _normalize_provider_order([])


def test_normalize_provider_order_dedup() -> None:
    result = _normalize_provider_order(["ollama", "ollama", "zai"])
    assert result == ["ollama", "zai"]


def test_normalize_provider_order_case_insensitive() -> None:
    result = _normalize_provider_order(["OLLAMA", "Zai"])
    assert result == ["ollama", "zai"]


def test_model_blocks(tmp_path: Path) -> None:
    models_file = _models_file(tmp_path)
    blocks = _model_blocks(models_file)
    assert len(blocks) == 3
    assert blocks[0].name == "model-a"


def test_model_blocks_empty(tmp_path: Path) -> None:
    models_file = tmp_path / "models.yaml"
    models_file.write_text("models: []\n")
    blocks = _model_blocks(models_file)
    assert len(blocks) == 0


def test_strip_yaml_scalar_plain() -> None:
    assert _strip_yaml_scalar("hello") == "hello"


def test_strip_yaml_scalar_single_quotes() -> None:
    assert _strip_yaml_scalar("'hello'") == "hello"


def test_strip_yaml_scalar_double_quotes() -> None:
    assert _strip_yaml_scalar('"hello"') == "hello"


def test_line_indent_spaces() -> None:
    assert _line_indent("    text") == "    "


def test_line_indent_no_indent() -> None:
    assert _line_indent("text") == ""


def test_table_exists_yes(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER)")
    assert _table_exists(conn, "test_table") is True


def test_table_exists_no(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        pass
    assert _table_exists(conn, "nonexistent") is False


def test_format_mapping() -> None:
    result = _format_mapping({"a": 1, "b": 2})
    assert "a=1" in result
    assert "b=2" in result


def test_format_mapping_not_dict() -> None:
    assert _format_mapping("not dict") == ""


def test_format_mapping_empty() -> None:
    assert _format_mapping({}) == ""


def test_read_log_tail_nonexistent() -> None:
    assert _read_log_tail("/nonexistent/path.log", 10) is None


def test_read_log_tail_short(tmp_path: Path) -> None:
    log = tmp_path / "test.log"
    log.write_text("line1\nline2\n")
    result = _read_log_tail(str(log), 5)
    assert "line1" in result
    assert "line2" in result


def test_read_log_tail_truncate(tmp_path: Path) -> None:
    log = tmp_path / "test.log"
    lines = [f"line{i}" for i in range(20)]
    log.write_text("\n".join(lines) + "\n")
    result = _read_log_tail(str(log), 5)
    assert "line19" in result
    assert "line0" not in result


def test_journalctl_follow_command() -> None:
    cmd = _journalctl_follow_command("testunit", 50)
    assert "journalctl" in cmd[0]
    assert "-u" in cmd
    assert "testunit" in cmd
    assert "-f" in cmd


def test_journalctl_available() -> None:
    # Just test it doesn't crash
    _journalctl_available()


def test_follow_log_file_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    follow_log_file("/nonexistent/path.log")
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_follow_log_file_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "empty.log"
    log.write_text("")
    # This will block forever, so we need to test differently
    # Just check initial output by reading
    content = _read_log_tail(str(log), 25)
    assert content is None or content == ""