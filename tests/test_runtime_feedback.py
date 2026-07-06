"""Tests for feedback loop, collector, runtime, memory, and config modules."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from llmrouter.config import (
    Settings,
    ServerConfig,
    ProvidersConfig,
    ProviderConfig,
    RoutingConfig,
    EvaluatorConfig,
    HealthConfig,
    MemoryConfig,
    PrecogConfig,
    LogLevel,
    get_settings,
    reload_settings,
    _load_yaml,
)
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ModelInfo, Provider, Tier, RoutingGrade
from llmrouter.evaluator.collector import ObservationCollector, _row_to_observation
from llmrouter.evaluator.feedback import FeedbackLoop, FeedbackReport
from llmrouter.evaluator.grader import RoutingDecisionGrader
from llmrouter.evaluator.judge import QualityJudge
from llmrouter.evaluator.types import QualityScore, RoutingObservation, RoutingReview, TrainingExample
from llmrouter.memory import (
    MemoryConfig as MemConfig,
    MemoryEntry,
    SQLiteMemoryStore,
    PrecogMemoryStore,
    PrecogMemoryConfig,
    render_memory_context,
    _token_weights,
    _cosine_score,
    _json_dict,
    _compact,
    _precog_entry,
)
from llmrouter.runtime import (
    build_app,
    build_providers,
    build_registry,
    _build_health_tracker,
    _build_memory_store,
    _ensure_runtime_logging,
    _is_insufficient_balance_error,
    _scorer_weights,
)
from llmrouter.providers.base import ProviderError
from llmrouter.utils import resolve_api_key


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_load_yaml_nonexistent() -> None:
    assert _load_yaml(Path("/nonexistent/file.yaml")) == {}


def test_load_yaml_valid(tmp_path: Path) -> None:
    f = tmp_path / "test.yaml"
    f.write_text("key: value\n")
    assert _load_yaml(f) == {"key": "value"}


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "LLMrouter"
    assert settings.debug is False
    assert settings.log_level == LogLevel.INFO


def test_reload_settings() -> None:
    s1 = get_settings()
    s2 = reload_settings()
    assert s1.app_name == s2.app_name


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def test_is_insufficient_balance_error_402() -> None:
    exc = ProviderError("insufficient balance", status_code=402)
    assert _is_insufficient_balance_error(exc) is True


def test_is_insufficient_balance_error_429() -> None:
    exc = ProviderError("quota exceeded", status_code=429)
    assert _is_insufficient_balance_error(exc) is True


def test_is_insufficient_balance_error_500() -> None:
    exc = ProviderError("server error", status_code=500)
    assert _is_insufficient_balance_error(exc) is False


def test_is_insufficient_balance_error_chinese() -> None:
    exc = ProviderError("余额不足", status_code=402)
    assert _is_insufficient_balance_error(exc) is True


def test_scorer_weights() -> None:
    from llmrouter.core.scorer import ScorerWeights
    weights = _scorer_weights({"length": 0.5, "code_detection": 0.5})
    assert weights.length == 0.5
    assert weights.code_detection == 0.5


def test_scorer_weights_defaults() -> None:
    from llmrouter.core.scorer import ScorerWeights
    weights = _scorer_weights({})
    assert weights.length == 0.15


def test_ensure_runtime_logging() -> None:
    _ensure_runtime_logging(debug=True)
    import logging
    logger = logging.getLogger("llmrouter")
    assert logger.level == logging.DEBUG


def test_build_registry_nonexistent() -> None:
    reg = build_registry("/nonexistent/path/to/models.yaml")
    assert len(reg.models) == 0


def test_build_health_tracker_memory() -> None:
    settings = Settings()
    settings.health.backend = "memory"
    tracker = _build_health_tracker(settings)
    assert tracker is not None


def test_build_health_tracker_sqlite(tmp_path: Path) -> None:
    settings = Settings()
    settings.health.backend = "sqlite"
    settings.health.db_path = str(tmp_path / "health.db")
    tracker = _build_health_tracker(settings)
    assert tracker is not None


def test_build_health_tracker_redis_fallback() -> None:
    settings = Settings()
    settings.health.backend = "redis"
    tracker = _build_health_tracker(settings)
    assert tracker is not None


def test_build_memory_store_disabled() -> None:
    settings = Settings()
    settings.memory.enabled = False
    assert _build_memory_store(settings) is None


def test_build_memory_store_sqlite() -> None:
    settings = Settings()
    settings.memory.enabled = True
    settings.memory.backend = "local"
    store = _build_memory_store(settings)
    assert store is not None
    assert isinstance(store, SQLiteMemoryStore)


def test_build_memory_store_precog() -> None:
    settings = Settings()
    settings.memory.enabled = True
    settings.memory.backend = "precog"
    settings.precog.enabled = True
    store = _build_memory_store(settings)
    assert store is not None
    assert isinstance(store, PrecogMemoryStore)


def test_build_providers_empty() -> None:
    settings = Settings()
    reg = ModelRegistry()
    providers = build_providers(settings, reg)
    assert len(providers) == 0


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_from_config() -> None:
    config = ProviderConfig(api_key="direct-key")
    assert resolve_api_key(config) == "direct-key"


def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "env-key")
    config = ProviderConfig(api_key=None)
    assert resolve_api_key(config, "TEST_API_KEY") == "env-key"


def test_resolve_api_key_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
    config = ProviderConfig(api_key=None)
    assert resolve_api_key(config, "NONEXISTENT_KEY") is None


# ---------------------------------------------------------------------------
# ObservationCollector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_record_and_flush(tmp_path: Path) -> None:
    collector = ObservationCollector(db_path=str(tmp_path / "test.db"), buffer_size=10)
    obs = RoutingObservation(
        prompt="test prompt",
        chosen_model="model-a",
        response="test response",
        latency_ms=100.0,
        cost_usd=0.01,
        prompt_tokens=10,
        completion_tokens=5,
    )
    collector.record(obs)
    ids = await collector.flush()
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_collector_record_with_sample_rate(tmp_path: Path) -> None:
    collector = ObservationCollector(db_path=str(tmp_path / "test.db"), buffer_size=10, sample_rate=0.0)
    obs = RoutingObservation(prompt="p", chosen_model="m", response="r", latency_ms=1)
    collector.record(obs)
    ids = await collector.flush()
    assert len(ids) == 0  # Sample rate 0 = nothing recorded


@pytest.mark.asyncio
async def test_collector_save_and_get_review(tmp_path: Path) -> None:
    collector = ObservationCollector(db_path=str(tmp_path / "test.db"))
    obs = RoutingObservation(prompt="p", chosen_model="m", response="r", latency_ms=1)
    collector.record(obs)
    ids = await collector.flush()
    assert len(ids) == 1

    review = RoutingReview(
        observation_id=ids[0],
        quality=QualityScore(4, 4, 4, 4, 4),
        grade=RoutingGrade.OPTIMAL,
    )
    await collector.save_review(review)

    pending = await collector.get_pending_observations()
    assert len(pending) == 0  # No pending after review


@pytest.mark.asyncio
async def test_collector_get_training_data(tmp_path: Path) -> None:
    collector = ObservationCollector(db_path=str(tmp_path / "test.db"))
    obs = RoutingObservation(prompt="p", chosen_model="m", response="r", latency_ms=1)
    collector.record(obs)
    ids = await collector.flush()

    review = RoutingReview(
        observation_id=ids[0],
        quality=QualityScore(4, 4, 4, 4, 4),
        grade=RoutingGrade.OPTIMAL,
    )
    await collector.save_review(review)

    training = await collector.get_training_data()
    assert len(training) == 1
    assert training[0].grade == RoutingGrade.OPTIMAL


def test_row_to_observation() -> None:
    row = ("prompt", "model", "response", 100.0, 0.01, 10, 5, 0.5, 2, '{"key": "value"}')
    obs = _row_to_observation(row)
    assert obs.prompt == "prompt"
    assert obs.chosen_model == "model"
    assert obs.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# FeedbackLoop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_loop_run_cycle(tmp_path: Path) -> None:
    collector = ObservationCollector(db_path=str(tmp_path / "test.db"))

    # Record observation
    obs = RoutingObservation(prompt="hello world", chosen_model="cheap", response="ok", latency_ms=100)
    collector.record(obs)
    await collector.flush()

    # Mock judge
    class MockJudge:
        async def evaluate(self, prompt: str, response: str, model: str) -> QualityScore:
            return QualityScore(5, 5, 5, 5, 5, "great")

    registry = ModelRegistry(
        models=(ModelInfo(name="cheap", provider=Provider.OPENAI, tier=Tier.T1),)
    )
    loop = FeedbackLoop(collector, MockJudge(), RoutingDecisionGrader(), registry)
    report = await loop.run_cycle()
    assert report.evaluated == 1
    assert report.optimal == 1


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_token_weights() -> None:
    weights = _token_weights("hello world hello")
    assert weights["hello"] == 2.0
    assert weights["world"] == 1.0


def test_cosine_score_identical() -> None:
    w = {"a": 1.0, "b": 2.0}
    assert _cosine_score(w, w) == pytest.approx(1.0)


def test_cosine_score_empty() -> None:
    assert _cosine_score({}, {"a": 1.0}) == 0.0


def test_cosine_score_no_overlap() -> None:
    assert _cosine_score({"a": 1.0}, {"b": 1.0}) == 0.0


def test_json_dict_valid() -> None:
    assert _json_dict('{"a": 1}') == {"a": 1}


def test_json_dict_invalid() -> None:
    assert _json_dict("not json") == {}


def test_json_dict_none() -> None:
    assert _json_dict(None) == {}


def test_compact_short() -> None:
    assert _compact("hello", 10) == "hello"


def test_compact_long() -> None:
    result = _compact("a" * 100, 10)
    assert len(result) == 10
    assert result.endswith("…")


def test_compact_limit_1() -> None:
    assert _compact("hello", 1) == "h"


def test_render_memory_context_empty() -> None:
    assert render_memory_context([], max_chars=100) == ""


def test_render_memory_context_zero_chars() -> None:
    entry = MemoryEntry(id=1, project="p", prompt="prompt", response="response", score=0.5)
    assert render_memory_context([entry], max_chars=0) == ""


def test_render_memory_context_normal() -> None:
    entry = MemoryEntry(id=1, project="p", prompt="prompt text", response="response text", score=0.9)
    result = render_memory_context([entry], max_chars=500)
    assert "Relevant project memory" in result
    assert "score=0.90" in result


def test_precog_entry_valid() -> None:
    item = {"id": 1, "prompt": "p", "response": "r", "score": 0.8}
    entry = _precog_entry(item, project="proj", fallback_id=99)
    assert entry is not None
    assert entry.id == 1
    assert entry.score == 0.8


def test_precog_entry_empty() -> None:
    item = {"prompt": "", "response": ""}
    entry = _precog_entry(item, project="p", fallback_id=1)
    assert entry is None


def test_precog_entry_invalid_score() -> None:
    item = {"prompt": "p", "response": "r", "score": "bad"}
    entry = _precog_entry(item, project="p", fallback_id=1)
    assert entry is not None
    assert entry.score == 0.0


def test_precog_entry_fallback_id() -> None:
    item = {"prompt": "p", "response": "r", "id": "not-a-number"}
    entry = _precog_entry(item, project="p", fallback_id=42)
    assert entry is not None
    assert entry.id == 42


def test_sqlite_memory_store_record_and_retrieve(tmp_path: Path) -> None:
    config = MemConfig(
        enabled=True,
        db_path=str(tmp_path / "memory.db"),
        min_prompt_chars=5,
        min_response_chars=5,
    )
    store = SQLiteMemoryStore(config)
    recorded = store.record_interaction(
        project="test",
        prompt="This is a test prompt for memory",
        response="This is a test response for memory",
        metadata={"key": "value"},
    )
    assert recorded is True

    entries = store.retrieve(project="test", query="test prompt memory")
    assert len(entries) > 0


def test_sqlite_memory_store_skip_short(tmp_path: Path) -> None:
    config = MemConfig(
        enabled=True,
        db_path=str(tmp_path / "memory.db"),
        min_prompt_chars=80,
        min_response_chars=40,
    )
    store = SQLiteMemoryStore(config)
    recorded = store.record_interaction(project="test", prompt="short", response="short")
    assert recorded is False


def test_sqlite_memory_store_disabled() -> None:
    config = MemConfig(enabled=False)
    store = SQLiteMemoryStore(config)
    assert store.retrieve(project="p", query="q") == []
    assert store.record_interaction(project="p", prompt="p", response="r") is False