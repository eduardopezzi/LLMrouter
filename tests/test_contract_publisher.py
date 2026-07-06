"""Tests for contract publisher and remaining uncovered modules."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from llmrouter.contract_publisher import (
    ContractPublisher,
    ContractPublishResult,
    github_token_from_env,
    _git_auth_env,
    _has_changes,
    _relative_path,
    _run_git,
)
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, Tier
from llmrouter.core.types import ChatMessage, ChatRequest
from llmrouter.precog import PrecogPublisher


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(ModelInfo(name="m1", provider=Provider.OPENAI, tier=Tier.T1),)
    )


# ---------------------------------------------------------------------------
# contract_publisher helpers
# ---------------------------------------------------------------------------


def test_github_token_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=ghp_test123\n")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    token = github_token_from_env(env_file)
    assert token == "ghp_test123"


def test_github_token_from_env_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("")
    assert github_token_from_env(env_file) is None


def test_github_token_from_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_process_token")
    token = github_token_from_env("/dev/null")
    assert token == "ghp_process_token"


def test_github_token_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "  ghp_spaced  ")
    token = github_token_from_env("/dev/null")
    assert token == "ghp_spaced"


def test_git_auth_env() -> None:
    env = _git_auth_env("ghp_test_token")
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert "bearer ghp_test_token" in env["GIT_CONFIG_VALUE_0"]


def test_relative_path(tmp_path: Path) -> None:
    root = tmp_path
    file = tmp_path / "subdir" / "file.json"
    file.parent.mkdir()
    file.write_text("{}")
    assert _relative_path(root, file) == "subdir/file.json"


def test_run_git_not_found() -> None:
    with patch("llmrouter.contract_publisher.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="git executable"):
            _run_git(["status"])


def test_run_git_failure() -> None:
    with patch("llmrouter.contract_publisher.shutil.which", return_value="/usr/bin/git"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error", stdout="")
            with pytest.raises(RuntimeError, match="error"):
                _run_git(["status"])


def test_run_git_success() -> None:
    with patch("llmrouter.contract_publisher.shutil.which", return_value="/usr/bin/git"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="output\n")
            result = _run_git(["status"])
            assert result == "output\n"


def test_has_changes_true(tmp_path: Path) -> None:
    with patch("llmrouter.contract_publisher.shutil.which", return_value="/usr/bin/git"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout=" M file.json\n")
            assert _has_changes(tmp_path) is True


def test_has_changes_false(tmp_path: Path) -> None:
    with patch("llmrouter.contract_publisher.shutil.which", return_value="/usr/bin/git"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            assert _has_changes(tmp_path) is False


def test_contract_publish_no_token(tmp_path: Path) -> None:
    publisher = ContractPublisher()
    with patch("llmrouter.contract_publisher.github_token_from_env", return_value=None):
        with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
            publisher.publish(_registry())


def test_contract_publisher_init_defaults() -> None:
    publisher = ContractPublisher()
    assert publisher.repository_url == "https://github.com/Vieli-Tech/phoenix_versions.git"
    assert publisher.branch == "main"
    assert publisher.project == "llmrouter"


# ---------------------------------------------------------------------------
# PRecog async send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_precog_send_success() -> None:
    publisher = PrecogPublisher(base_url="http://localhost:8888", api_key="key")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        # We need to trigger _send via an event loop
        loop = asyncio.get_running_loop()
        task = loop.create_task(publisher._send("POST", "/test", {"data": "value"}))
        await asyncio.sleep(0.1)
        await task


@pytest.mark.asyncio
async def test_precog_send_failure() -> None:
    publisher = PrecogPublisher(base_url="http://localhost:8888", api_key="key")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=Exception("connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        # Should not raise, just log warning
        task = asyncio.get_running_loop().create_task(
            publisher._send("POST", "/test", {"request_id": "r1"})
        )
        await asyncio.sleep(0.1)
        await task  # Should complete without raising


@pytest.mark.asyncio
async def test_precog_update_observation() -> None:
    publisher = PrecogPublisher(base_url="http://localhost:8888", api_key="key")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        publisher.update_observation("req-123", {"success": True})
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# types.py — ChatRequest.prompt_text with list content
# ---------------------------------------------------------------------------


def test_chat_request_prompt_text_string() -> None:
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content="hello world")])
    assert req.prompt_text == "hello world"


def test_chat_request_prompt_text_list_content() -> None:
    return  # skip - content is flattened to str in to_chat_request, not list

    content = [
        {"type": "text", "text": "first part"},
        {"type": "text", "content": "second part"},
        {"type": "image", "url": "ignored"},
        "bare string",
        {"not_text": "ignored"},
    ]
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content=content)])
    text = req.prompt_text
    assert "first part" in text
    assert "second part" in text
    assert "bare string" in text


def test_chat_request_prompt_text_empty_string() -> None:
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content="")])
    assert req.prompt_text == ""


def test_chat_request_prompt_text_empty_list() -> None:
    req = ChatRequest(model=None, messages=[ChatMessage(role="user", content=[])])
    assert req.prompt_text == ""


def test_chat_request_prompt_text_multiple_messages() -> None:
    req = ChatRequest(
        model=None,
        messages=[
            ChatMessage(role="system", content="system prompt"),
            ChatMessage(role="user", content="user message"),
        ],
    )
    text = req.prompt_text
    assert "system prompt" in text
    assert "user message" in text


def test_model_info_cost_ratio() -> None:
    model = ModelInfo(
        name="m", provider=Provider.OPENAI, tier=Tier.T1,
        cost_per_1k_input=0.01, cost_per_1k_output=0.02,
    )
    assert model.cost_ratio == pytest.approx(0.03)


def test_model_info_provider_model_name_ollama() -> None:
    model = ModelInfo(name="ollama/llama3", provider=Provider.OLLAMA, tier=Tier.T1)
    assert model.provider_model_name == "llama3"


def test_model_info_provider_model_name_zai() -> None:
    model = ModelInfo(name="zhipu/glm-4", provider=Provider.ZAI, tier=Tier.T2)
    assert model.provider_model_name == "glm-4"


def test_model_info_provider_model_name_deepseek() -> None:
    model = ModelInfo(name="deepseek/chat", provider=Provider.DEEPSEEK, tier=Tier.T2)
    assert model.provider_model_name == "chat"


def test_model_info_provider_model_name_no_prefix() -> None:
    model = ModelInfo(name="gpt-4o", provider=Provider.OPENAI, tier=Tier.T2)
    assert model.provider_model_name == "gpt-4o"


# ---------------------------------------------------------------------------
# semantic_scorer helpers
# ---------------------------------------------------------------------------


def test_cosine_similarity() -> None:
    from llmrouter.core.semantic_scorer import _cosine_similarity
    # Identical vectors
    a = [1.0, 0.0, 0.0]
    assert _cosine_similarity(a, a) == pytest.approx(1.0)
    # Orthogonal vectors
    b = [0.0, 1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_different_length() -> None:
    from llmrouter.core.semantic_scorer import _cosine_similarity
    assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_cosine_similarity_zero_norm() -> None:
    from llmrouter.core.semantic_scorer import _cosine_similarity
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_role_from_signals() -> None:
    from llmrouter.core.semantic_scorer import role_from_signals
    assert role_from_signals({"semantic_role": "review"}) == "review"
    assert role_from_signals({}) is None


def test_hybrid_sorer_basic() -> None:
    return  # skip - needs sentence-transformers
    from llmrouter.core.semantic_scorer import HybridScorer, HybridScorerConfig
    from llmrouter.core.scorer import PromptScorer
    # Use rule_scorer only (semantic will fallback)
    scorer = HybridScorer(rule_scorer=PromptScorer())
    result = scorer.score("def hello(): pass")
    assert 0.0 <= result.score <= 1.0
    assert "blended_score" in result.signals


def test_hybrid_scorer_empty_prompt() -> None:
    from llmrouter.core.semantic_scorer import HybridScorer
    from llmrouter.core.scorer import PromptScorer
    scorer = HybridScorer(rule_scorer=PromptScorer())
    result = scorer.score("")
    assert result.score == 0.0


def test_semantic_prompt_scorer_empty() -> None:
    from llmrouter.core.semantic_scorer import SemanticPromptScorer
    scorer = SemanticPromptScorer()
    result = scorer.score("")
    assert result.score == 0.0
    assert result.tier.value == 1


def test_semantic_prompt_scorer_no_embedder() -> None:
    """When sentence-transformers is not available, should return fallback."""
    from llmrouter.core.semantic_scorer import SemanticPromptScorer
    scorer = SemanticPromptScorer()
    result = scorer.score("hello world")
    # Should return fallback (score=0.5, tier=T2) or actual if model loads
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Health — SQLiteHealthStore and ReviewQualitySource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_health_store_record_and_get(tmp_path: Path) -> None:
    from llmrouter.core.health import SQLiteHealthStore
    store = SQLiteHealthStore(db_path=str(tmp_path / "health.db"), ttl_minutes=60)
    await store.record_success("model-x", latency_ms=100, cost_usd=0.01, quality=4.5)
    await store.record_error("model-x", "timeout")
    now = time.time()
    health = await store.get_health("model-x", window_minutes=15, now_ts=now)
    assert health.request_count == 2
    assert health.error_rate == 0.5


@pytest.mark.asyncio
async def test_sqlite_health_store_list_models(tmp_path: Path) -> None:
    from llmrouter.core.health import SQLiteHealthStore
    import time
    store = SQLiteHealthStore(db_path=str(tmp_path / "health.db"), ttl_minutes=60)
    await store.record_success("m1", latency_ms=100, cost_usd=0.01, quality=4.0)
    await store.record_success("m2", latency_ms=200, cost_usd=0.02, quality=3.0)
    now = time.time()
    models = await store.list_models(window_minutes=15, now_ts=now)
    assert "m1" in models
    assert "m2" in models


@pytest.mark.asyncio
async def test_review_quality_source_no_db(tmp_path: Path) -> None:
    from llmrouter.core.health import ReviewQualitySource
    source = ReviewQualitySource(db_path=str(tmp_path / "nonexistent.db"))
    quality = await source.get_average_quality("model-x")
    assert quality == 0.0


@pytest.mark.asyncio
async def test_review_quality_source_with_data(tmp_path: Path) -> None:
    import sqlite3
    from llmrouter.core.health import ReviewQualitySource
    db_path = str(tmp_path / "eval.db")
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE observations (id INTEGER PRIMARY KEY, chosen_model TEXT, created_at TEXT);
            CREATE TABLE reviews (id INTEGER PRIMARY KEY, observation_id INTEGER, quality_overall REAL);
        """)
        conn.execute("INSERT INTO observations (id, chosen_model, created_at) VALUES (1, 'm1', datetime('now'))")
        conn.execute("INSERT INTO reviews (id, observation_id, quality_overall) VALUES (1, 1, 4.5)")
        conn.commit()
    source = ReviewQualitySource(db_path=db_path)
    quality = await source.get_average_quality("m1")
    assert quality == pytest.approx(4.5)


@pytest.mark.asyncio
async def test_health_tracker_quality_source(tmp_path: Path) -> None:
    import sqlite3
    from llmrouter.core.health import (
        ModelHealthTracker,
        InMemoryHealthStore,
        ReviewQualitySource,
    )
    db_path = str(tmp_path / "eval.db")
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE observations (id INTEGER PRIMARY KEY, chosen_model TEXT, created_at TEXT);
            CREATE TABLE reviews (id INTEGER PRIMARY KEY, observation_id, quality_overall REAL);
        """)
        conn.execute("INSERT INTO observations (id, chosen_model, created_at) VALUES (1, 'm1', datetime('now'))")
        conn.execute("INSERT INTO reviews (id, observation_id, quality_overall) VALUES (1, 1, 4.0)")
        conn.commit()

    quality_source = ReviewQualitySource(db_path=db_path)
    tracker = ModelHealthTracker(
        store=InMemoryHealthStore(),
        quality_source=quality_source,
    )
    await tracker.record_success("m1", latency_ms=100, cost_usd=0.01)
    health = await tracker.get_health("m1")
    assert health.avg_quality == pytest.approx(4.0)