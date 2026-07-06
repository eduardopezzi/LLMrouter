"""Tests for remaining providers, logging, precog, and cross-repository modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmrouter.core.types import ChatRequest, ChatMessage, Provider, Tier, ModelInfo
from llmrouter.cross_repository import (
    BreakingChangeDetector,
    ContractChange,
    ContractCheckResult,
    ContractRegistry,
    ChangeSeverity,
    _by_key,
    _string_list,
    _find_project_dir,
    format_contract_changes,
    load_contract_snapshot,
    resolve_project_contract_path,
)
from llmrouter.core.registry import ModelRegistry
from llmrouter.logging_config import ColoredFormatter, get_logger, setup_logging
from llmrouter.precog import PrecogPublisher
from llmrouter.providers.gemini_provider import GeminiProvider
from llmrouter.providers.openai_provider import OpenAIProvider
from llmrouter.providers.zai_provider import ZaiProvider


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_provider_chat_not_implemented() -> None:
    provider = GeminiProvider(api_key="key")
    request = ChatRequest(model="gemini-pro", messages=[ChatMessage(role="user", content="hi")])
    from llmrouter.providers.base import ProviderError
    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(request, "gemini-pro")
    assert exc_info.value.status_code == 501


@pytest.mark.asyncio
async def test_gemini_provider_stream_not_implemented() -> None:
    provider = GeminiProvider(api_key="key")
    request = ChatRequest(model="gemini-pro", messages=[ChatMessage(role="user", content="hi")])
    from llmrouter.providers.base import ProviderError
    with pytest.raises(ProviderError):
        async for _ in provider.stream_completion(request, "gemini-pro"):
            pass


def test_openai_provider_init() -> None:
    provider = OpenAIProvider(api_key="sk-test")
    assert provider.name == "openai"
    assert provider._api_key == "sk-test"
    headers = provider._build_headers()
    assert headers["Authorization"] == "Bearer sk-test"


def test_openai_provider_default_base_url() -> None:
    provider = OpenAIProvider()
    assert provider._base_url == "https://api.openai.com/v1"


def test_zai_provider_init() -> None:
    provider = ZaiProvider(api_key="zai-key")
    assert provider.name == "zai"
    assert provider._api_key == "zai-key"
    headers = provider._build_headers()
    assert headers["Authorization"] == "Bearer zai-key"
    assert headers["Accept-Language"] == "en-US,en"


def test_zai_provider_default_base_url() -> None:
    provider = ZaiProvider()
    assert provider._base_url == "https://api.z.ai/api/paas/v4"


def test_ollama_provider_url_normalization() -> None:
    from llmrouter.providers.ollama_provider import OllamaProvider
    provider = OllamaProvider(base_url="http://localhost:11434")
    assert provider._base_url == "http://localhost:11434/v1"

    provider2 = OllamaProvider(base_url="http://localhost:11434/v1")
    assert provider2._base_url == "http://localhost:11434/v1"

    provider3 = OllamaProvider(base_url="http://localhost:11434/")
    assert provider3._base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_get_logger() -> None:
    logger = get_logger("test.module")
    assert logger.name == "test.module"


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    """Restore logging configuration after each test in this module."""
    import logging
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    yield
    root.setLevel(original_level)
    root.handlers = original_handlers


def test_setup_logging_debug() -> None:
    import logging
    setup_logging(debug=True)
    # Root logger should be at DEBUG level
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_info() -> None:
    setup_logging(debug=False)
    # Root logger should be at INFO
    import logging
    assert logging.getLogger().level == 20  # INFO


def test_colored_formatter() -> None:
    import logging
    formatter = ColoredFormatter("[%(levelname)s] %(message)s")
    record = logging.LogRecord(
        name="test", level=logging.DEBUG, pathname="", lineno=0,
        msg="test message", args=(), exc_info=None,
    )
    result = formatter.format(record)
    assert "test message" in result
    assert "\033[36m" in result  # Cyan for DEBUG


# ---------------------------------------------------------------------------
# PRecog
# ---------------------------------------------------------------------------


def test_precog_publisher_init() -> None:
    publisher = PrecogPublisher(base_url="http://localhost:8888", api_key="key")
    assert publisher._base_url == "http://localhost:8888"
    assert publisher._api_key == "key"


def test_precog_publisher_no_loop_silently_drops() -> None:
    """record_observation without running loop should not raise."""
    publisher = PrecogPublisher(base_url="http://localhost:8888")
    # No running event loop → should silently return
    publisher.record_observation({"request_id": "test"})


# ---------------------------------------------------------------------------
# Cross-repository
# ---------------------------------------------------------------------------


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=(
            ModelInfo(name="model-a", provider=Provider.OPENAI, tier=Tier.T2, capabilities=frozenset({"code", "review"})),
            ModelInfo(name="model-b", provider=Provider.OLLAMA, tier=Tier.T1, capabilities=frozenset({"summarization"})),
        )
    )


def test_contract_snapshot() -> None:
    registry = _registry()
    contract = ContractRegistry(registry=registry, service_name="test-service")
    snap = contract.snapshot()
    assert snap["service"] == "test-service"
    assert snap["schema_version"] == "1.0"
    assert len(snap["models"]) == 2
    assert "code" in snap["routing_roles"]
    assert "review" in snap["routing_roles"]
    assert "summarization" in snap["routing_roles"]


def test_contract_write_snapshot(tmp_path: Path) -> None:
    registry = _registry()
    contract = ContractRegistry(registry=registry)
    output = tmp_path / "contract.json"
    contract.write_snapshot(output)
    assert output.exists()
    data = json.loads(output.read_text())
    assert "models" in data


def test_breaking_change_no_changes() -> None:
    detector = BreakingChangeDetector()
    snap = {"schema_version": "1.0", "endpoints": [], "models": [], "routing_roles": []}
    result = detector.compare(snap, snap)
    assert result.is_compatible
    assert len(result.changes) == 0


def test_breaking_change_schema_version() -> None:
    detector = BreakingChangeDetector()
    result = detector.compare(
        {"schema_version": "1.0"},
        {"schema_version": "2.0"},
    )
    assert not result.is_compatible
    assert len(result.breaking_changes) == 1


def test_breaking_change_endpoint_removed() -> None:
    detector = BreakingChangeDetector()
    before = {"endpoints": [{"path": "/v1/chat", "method": "POST", "request_schema": {}}]}
    after = {"endpoints": []}
    result = detector.compare(before, after)
    assert not result.is_compatible
    assert any("removed" in c.message for c in result.breaking_changes)


def test_breaking_change_endpoint_method_changed() -> None:
    detector = BreakingChangeDetector()
    before = {"endpoints": [{"path": "/v1/chat", "method": "POST"}]}
    after = {"endpoints": [{"path": "/v1/chat", "method": "GET"}]}
    result = detector.compare(before, after)
    assert not result.is_compatible


def test_non_breaking_change_endpoint_added() -> None:
    detector = BreakingChangeDetector()
    before = {"endpoints": []}
    after = {"endpoints": [{"path": "/v1/new", "method": "GET"}]}
    result = detector.compare(before, after)
    assert result.is_compatible
    assert len(result.non_breaking_changes) == 1


def test_breaking_change_model_removed() -> None:
    detector = BreakingChangeDetector()
    before = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": []}]}
    after = {"models": []}
    result = detector.compare(before, after)
    assert not result.is_compatible


def test_non_breaking_change_model_added() -> None:
    detector = BreakingChangeDetector()
    before = {"models": []}
    after = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": []}]}
    result = detector.compare(before, after)
    assert result.is_compatible


def test_breaking_change_model_provider_changed() -> None:
    detector = BreakingChangeDetector()
    before = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": []}]}
    after = {"models": [{"id": "m1", "provider": "ollama", "tier": 2, "capabilities": []}]}
    result = detector.compare(before, after)
    assert not result.is_compatible


def test_breaking_change_model_capabilities_removed() -> None:
    detector = BreakingChangeDetector()
    before = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": ["code", "review"]}]}
    after = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": ["code"]}]}
    result = detector.compare(before, after)
    assert not result.is_compatible


def test_non_breaking_change_model_capabilities_added() -> None:
    detector = BreakingChangeDetector()
    before = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": ["code"]}]}
    after = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": ["code", "review"]}]}
    result = detector.compare(before, after)
    assert result.is_compatible


def test_breaking_change_context_window_decreased() -> None:
    detector = BreakingChangeDetector()
    before = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": [], "context_window": 8192}]}
    after = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": [], "context_window": 4096}]}
    result = detector.compare(before, after)
    assert not result.is_compatible


def test_non_breaking_change_context_window_increased() -> None:
    detector = BreakingChangeDetector()
    before = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": [], "context_window": 8192}]}
    after = {"models": [{"id": "m1", "provider": "openai", "tier": 2, "capabilities": [], "context_window": 16384}]}
    result = detector.compare(before, after)
    assert result.is_compatible


def test_breaking_change_routing_role_removed() -> None:
    detector = BreakingChangeDetector()
    before = {"routing_roles": ["code", "review"]}
    after = {"routing_roles": ["code"]}
    result = detector.compare(before, after)
    assert not result.is_compatible


def test_non_breaking_change_routing_role_added() -> None:
    detector = BreakingChangeDetector()
    before = {"routing_roles": ["code"]}
    after = {"routing_roles": ["code", "review"]}
    result = detector.compare(before, after)
    assert result.is_compatible


def test_format_contract_changes_no_changes() -> None:
    result = ContractCheckResult()
    assert format_contract_changes(result) == "No contract changes detected."


def test_format_contract_changes_with_changes() -> None:
    changes = (ContractChange(ChangeSeverity.BREAKING, "models.m1", "Model removed"),)
    result = ContractCheckResult(changes=changes)
    formatted = format_contract_changes(result)
    assert "breaking: models.m1" in formatted
    assert "Model removed" in formatted


def test_by_key() -> None:
    data = [{"path": "/a"}, {"path": "/b"}, {"not_path": "x"}]
    result = _by_key(data, "path")
    assert "/a" in result
    assert "/b" in result
    assert len(result) == 2


def test_by_key_not_list() -> None:
    assert _by_key("not a list", "path") == {}


def test_string_list() -> None:
    assert _string_list(["a", "b"]) == ["a", "b"]
    assert _string_list([1, "b"]) == ["b"]  # type: ignore[list-item]
    assert _string_list("not list") == []


def test_find_project_dir(tmp_path: Path) -> None:
    (tmp_path / "LLMrouter").mkdir()
    result = _find_project_dir(tmp_path, "llmrouter")
    assert result is not None
    assert result.name == "LLMrouter"


def test_find_project_dir_not_found(tmp_path: Path) -> None:
    assert _find_project_dir(tmp_path, "nonexistent") is None


def test_find_project_dir_root_not_exists() -> None:
    assert _find_project_dir(Path("/nonexistent/path"), "proj") is None


def test_resolve_project_contract_path_create(tmp_path: Path) -> None:
    result = resolve_project_contract_path(tmp_path, "newproj", "contract.json", create=True)
    assert result.parent.exists()
    assert result.name == "contract.json"


def test_resolve_project_contract_path_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_project_contract_path(tmp_path, "nonexistent", "c.json", create=False)


def test_resolve_project_contract_path_empty_project(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_project_contract_path(tmp_path, "", "c.json")


def test_resolve_project_contract_path_filename_with_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_project_contract_path(tmp_path, "proj", "dir/file.json")


def test_load_contract_snapshot(tmp_path: Path) -> None:
    snap_file = tmp_path / "snap.json"
    snap_file.write_text(json.dumps({"service": "test"}))
    result = load_contract_snapshot(snap_file)
    assert result["service"] == "test"


def test_load_contract_snapshot_not_dict(tmp_path: Path) -> None:
    snap_file = tmp_path / "snap.json"
    snap_file.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_contract_snapshot(snap_file)


def test_compare_files(tmp_path: Path) -> None:
    file1 = tmp_path / "before.json"
    file2 = tmp_path / "after.json"
    file1.write_text(json.dumps({"schema_version": "1.0", "models": []}))
    file2.write_text(json.dumps({"schema_version": "2.0", "models": []}))
    detector = BreakingChangeDetector()
    result = detector.compare_files(file1, file2)
    assert not result.is_compatible