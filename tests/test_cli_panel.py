from __future__ import annotations

import sqlite3

from llmrouter.cli_panel import (
    FALLBACK_COUNT_ENV,
    PROVIDER_COST_ORDER_ENV,
    ROUTING_STRATEGY_ENV,
    _journalctl_follow_command,
    _log_file_end_offset,
    _parse_provider_selection,
    _read_log_since,
    _read_log_tail,
    observation_stats,
    render_current_settings,
    render_panel_summary,
    set_fallback_count,
    set_provider_cost_order,
    set_routing_strategy,
)
from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, Tier


def test_panel_updates_env_file_preserving_existing_values(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("KEEP_ME=yes\n", encoding="utf-8")

    set_routing_strategy(env_path, "quality")
    set_fallback_count(env_path, 4)
    set_provider_cost_order(env_path, ["NVIDIA", "ZAI", "OLLAMA"])

    body = env_path.read_text(encoding="utf-8")
    assert "KEEP_ME=yes" in body
    assert f"{ROUTING_STRATEGY_ENV}=quality" in body
    assert f"{FALLBACK_COUNT_ENV}=4" in body
    assert f'{PROVIDER_COST_ORDER_ENV}=["nvidia", "zai", "ollama"]' in body


def test_panel_renders_catalog_and_empty_observation_stats(tmp_path) -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="nvidia/reviewer",
                provider=Provider.NVIDIA,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
            ),
        )
    )
    settings = Settings(evaluator={"db_path": str(tmp_path / "missing.db")})

    summary = render_panel_summary(settings, registry)

    assert "LLMrouter CLI Panel" in summary
    assert "strategy: cost" in summary
    assert "provider_cost_order: nvidia, zai, ollama" in summary
    assert "providers: nvidia=1" in summary
    assert "observations: 0" in summary


def test_observation_stats_reads_sqlite_database(tmp_path) -> None:
    db_path = tmp_path / "llmrouter.db"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                chosen_model TEXT NOT NULL,
                response TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                cost_usd REAL NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                scorer_score REAL,
                scorer_tier INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id INTEGER NOT NULL,
                relevance INTEGER NOT NULL,
                accuracy INTEGER NOT NULL,
                completeness INTEGER NOT NULL,
                concision INTEGER NOT NULL,
                safety INTEGER NOT NULL,
                quality_overall REAL NOT NULL,
                grade TEXT NOT NULL,
                suggested_model TEXT,
                rationale TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        db.execute(
            """
            INSERT INTO observations (
                prompt, chosen_model, response, latency_ms, cost_usd,
                prompt_tokens, completion_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("hello", "nvidia/reviewer", "hi", 25.0, 0.5, 10, 5),
        )
        db.execute(
            """
            INSERT INTO reviews (
                observation_id, relevance, accuracy, completeness, concision,
                safety, quality_overall, grade
            ) VALUES (1, 5, 5, 5, 5, 5, 5.0, 'optimal')
            """
        )
        db.commit()

    stats = observation_stats(db_path)

    assert stats["observations"] == 1
    assert stats["reviews"] == 1
    assert stats["avg_latency_ms"] == 25.0
    assert stats["total_cost_usd"] == 0.5
    assert stats["prompt_tokens"] == 10
    assert stats["completion_tokens"] == 5
    assert stats["top_models"] == [
        {"model": "nvidia/reviewer", "requests": 1, "avg_latency_ms": 25.0}
    ]


def test_render_current_settings_shows_all_sections(tmp_path) -> None:
    registry = ModelRegistry(
        models=(
            ModelInfo(
                name="nvidia/reviewer",
                provider=Provider.NVIDIA,
                tier=Tier.T3,
                capabilities=frozenset({"review"}),
            ),
        )
    )
    settings = Settings(evaluator={"db_path": str(tmp_path / "missing.db")})

    output = render_current_settings(settings, registry)

    assert "=== Current Settings ===" in output
    assert "Routing" in output
    assert "strategy:" in output
    assert "cost" in output
    assert "fallback_count:" in output
    assert "2" in output
    assert "provider_cost_order: nvidia, zai, ollama" in output
    assert "Scorer weights" in output
    assert "Server" in output
    assert "host: 0.0.0.0" in output
    assert "port: 12345" in output
    assert "Evaluator" in output
    assert "Debug" in output
    assert "Catalog summary" in output
    assert "providers_in_catalog: nvidia" in output


def test_parse_provider_selection_by_numbers() -> None:
    available = ["gemini", "nvidia", "ollama", "openai", "zai"]
    result = _parse_provider_selection("2,5,3", available)
    assert result == ["nvidia", "zai", "ollama"]


def test_parse_provider_selection_by_names() -> None:
    result = _parse_provider_selection("nvidia,zai,ollama", [])
    assert result == ["nvidia", "zai", "ollama"]


def test_parse_provider_selection_empty_returns_empty() -> None:
    assert _parse_provider_selection("", ["nvidia"]) == []
    assert _parse_provider_selection("   ", ["nvidia"]) == []


def test_parse_provider_selection_invalid_numbers_filtered() -> None:
    available = ["nvidia", "ollama"]
    result = _parse_provider_selection("1,9,2", available)
    assert result == ["nvidia", "ollama"]


def test_parse_provider_selection_deduplicates() -> None:
    available = ["nvidia", "ollama"]
    result = _parse_provider_selection("1,1,2,2", available)
    assert result == ["nvidia", "ollama"]


def test_read_log_tail_returns_none_when_missing() -> None:
    assert _read_log_tail("/nonexistent/path.log", 10) is None


def test_read_log_tail_returns_last_lines(tmp_path) -> None:
    log_path = tmp_path / "test.log"
    log_path.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    content = _read_log_tail(str(log_path), 2)
    assert content == "line4\nline5"

    content_all = _read_log_tail(str(log_path), 100)
    assert "line1" in content_all
    assert "line5" in content_all


def test_read_log_tail_handles_encoding_errors(tmp_path) -> None:
    log_path = tmp_path / "bad_encoding.log"
    log_path.write_bytes(b"valid line\n\xff\xfe bad bytes\nanother line\n")

    content = _read_log_tail(str(log_path), 10)
    assert content is not None
    assert "valid line" in content
    assert "another line" in content


def test_read_log_since_returns_appended_content(tmp_path) -> None:
    log_path = tmp_path / "follow.log"
    log_path.write_text("line1\nline2\n", encoding="utf-8")
    offset = _log_file_end_offset(log_path)

    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    chunk, new_offset = _read_log_since(log_path, offset)

    assert chunk == "line3\n"
    assert new_offset > offset


def test_read_log_since_handles_truncated_file(tmp_path) -> None:
    log_path = tmp_path / "follow.log"
    log_path.write_text("old line\nold line 2\n", encoding="utf-8")
    offset = _log_file_end_offset(log_path)

    log_path.write_text("new line\n", encoding="utf-8")
    chunk, new_offset = _read_log_since(log_path, offset)

    assert chunk == "new line\n"
    assert new_offset == len("new line\n")


def test_journalctl_follow_command_uses_last_25_lines_by_default() -> None:
    assert _journalctl_follow_command("llmrouter") == [
        "journalctl",
        "-u",
        "llmrouter",
        "-n",
        "25",
        "-f",
    ]
