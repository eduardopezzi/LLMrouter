from __future__ import annotations

from pathlib import Path

import pytest

from llmrouter.config import Settings
from llmrouter.runtime import _precog_memory_config, _resolve_precog_api_key


def test_precog_api_key_prefers_nested_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRECOG_INTERNAL_API_KEY", "internal-key")
    monkeypatch.setenv("LLMROUTER_OBSERVATION_API_KEY", "observation-key")
    settings = Settings()
    settings.precog.api_key = "nested-key"

    assert _resolve_precog_api_key(settings) == "nested-key"


def test_precog_api_key_accepts_precog_internal_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRECOG_INTERNAL_API_KEY", "internal-key")
    monkeypatch.delenv("LLMROUTER_OBSERVATION_API_KEY", raising=False)
    settings = Settings()
    settings.precog.api_key = None

    assert _resolve_precog_api_key(settings) == "internal-key"
    assert _precog_memory_config(settings).api_key == "internal-key"


def test_precog_api_key_accepts_observation_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRECOG_INTERNAL_API_KEY", raising=False)
    monkeypatch.setenv("LLMROUTER_OBSERVATION_API_KEY", "observation-key")
    settings = Settings()
    settings.precog.api_key = None

    assert _resolve_precog_api_key(settings) == "observation-key"


def test_precog_api_key_alias_loads_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PRECOG_INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("LLMROUTER_OBSERVATION_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "PRECOG_INTERNAL_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert _resolve_precog_api_key(settings) == "dotenv-key"
