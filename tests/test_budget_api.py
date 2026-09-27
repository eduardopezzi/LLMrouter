"""Tests for the budget enforcement (B2) and budget API (B3).

Covers: chat pre-flight check (soft warning header, hard 402), post-response
usage recording, tenant headers, GET/POST /v1/llmrouter/budgets, 503 without
manager, and opt-in config defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from llmrouter.api.routes import create_app
from llmrouter.config import Settings
from llmrouter.core.budget import BudgetLimits, BudgetManager
from llmrouter.core.types import ModelInfo, Provider, Tier
from tests.test_api import FakeProxy


def _registry() -> Any:
    from llmrouter.core.registry import ModelRegistry

    return ModelRegistry(
        models=(
            ModelInfo(
                name="cheap",
                provider=Provider.OPENAI,
                tier=Tier.T1,
                cost_per_1k_input=0.5,
                cost_per_1k_output=1.0,
            ),
        )
    )


def _manager(tmp_path: Path) -> BudgetManager:
    return BudgetManager(str(tmp_path / "budget.db"))


# ---------------------------------------------------------------------------
# Config defaults (opt-in)
# ---------------------------------------------------------------------------


def test_budget_config_defaults_are_opt_in() -> None:
    settings = Settings()

    assert settings.budgets.enabled is False
    assert settings.budgets.db_path == "data/budget.db"
    assert settings.budgets.hard_default_usd is None


# ---------------------------------------------------------------------------
# B3: GET/POST /v1/llmrouter/budgets
# ---------------------------------------------------------------------------


def test_budget_endpoints_503_without_manager() -> None:
    app = create_app(registry=_registry(), proxy=FakeProxy())
    client = TestClient(app)

    assert client.get("/v1/llmrouter/budgets/proj").status_code == 503
    assert (
        client.post(
            "/v1/llmrouter/budgets",
            json={"project_id": "proj"},
        ).status_code
        == 503
    )


def test_budget_post_then_get_persists_limits(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    app = create_app(registry=_registry(), proxy=FakeProxy(), budget_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/llmrouter/budgets",
        json={
            "project_id": "proj",
            "user_id": "alice",
            "daily_limit_usd": 1.5,
            "monthly_limit_usd": 20.0,
            "mode": "hard",
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True

    usage = client.get("/v1/llmrouter/budgets/proj", params={"user_id": "alice"})
    assert usage.status_code == 200
    body = usage.json()
    assert body["project_id"] == "proj"
    assert body["user_id"] == "alice"
    assert body["daily_limit_usd"] == 1.5
    assert body["monthly_limit_usd"] == 20.0
    assert body["mode"] == "hard"
    assert body["daily_spent_usd"] == 0.0


def test_budget_post_validates_payload() -> None:
    manager_case = None
    app = create_app(
        registry=_registry(),
        proxy=FakeProxy(),
        budget_manager=manager_case,
    )
    client = TestClient(app)

    # invalid mode -> 422
    invalid = client.post(
        "/v1/llmrouter/budgets",
        json={"project_id": "proj", "mode": "strict"},
    )
    assert invalid.status_code == 422
    # missing project_id -> 422
    missing = client.post("/v1/llmrouter/budgets", json={})
    assert missing.status_code == 422


# ---------------------------------------------------------------------------
# B2: chat enforcement
# ---------------------------------------------------------------------------


def test_chat_records_usage_after_response(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    app = create_app(registry=_registry(), proxy=FakeProxy(), budget_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "say hello"}]},
        headers={"X-Project-ID": "proj", "X-User-ID": "alice"},
    )
    assert response.status_code == 200

    usage = client.get("/v1/llmrouter/budgets/proj", params={"user_id": "alice"})
    # FakeProxy returns prompt_tokens=2, completion_tokens=1; model rates
    # 0.5/1.0 per 1k -> (2/1000)*0.5 + (1/1000)*1.0 = 0.002
    assert usage.json()["daily_spent_usd"] == 0.002


def test_chat_soft_mode_warning_header_and_200(tmp_path: Path) -> None:
    import asyncio

    manager = _manager(tmp_path)

    async def seed() -> None:
        await manager.set_limits(
            "proj",
            "alice",
            BudgetLimits(daily_limit_usd=0.001, mode="soft"),
        )
        # pre-flight uses estimated_cost=0.0, so the warning only fires once
        # recorded spend already exceeds the limit (0.002 > 0.001)
        await manager.record_usage("proj", "alice", 0.002)

    asyncio.run(seed())

    app = create_app(registry=_registry(), proxy=FakeProxy(), budget_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "say hello"}]},
        headers={"X-Project-ID": "proj", "X-User-ID": "alice"},
    )
    assert response.status_code == 200
    assert "X-Budget-Warning" in response.headers
    assert response.headers["X-Budget-Warning"]


def test_chat_hard_mode_blocks_with_402(tmp_path: Path) -> None:
    import asyncio

    manager = _manager(tmp_path)

    async def seed() -> None:
        await manager.set_limits(
            "proj",
            "alice",
            BudgetLimits(daily_limit_usd=0.001, mode="hard"),
        )
        # exceed the limit directly: 0.002 > 0.001
        await manager.record_usage("proj", "alice", 0.002)

    asyncio.run(seed())

    app = create_app(registry=_registry(), proxy=FakeProxy(), budget_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "say hello"}]},
        headers={"X-Project-ID": "proj", "X-User-ID": "alice"},
    )
    assert response.status_code == 402
    assert "daily budget exceeded" in response.json()["detail"]


def test_chat_missing_tenant_headers_fall_back_to_default(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    app = create_app(registry=_registry(), proxy=FakeProxy(), budget_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "say hello"}]},
    )
    assert response.status_code == 200

    usage = client.get("/v1/llmrouter/budgets/default", params={"user_id": "default"})
    assert usage.status_code == 200
    assert usage.json()["daily_spent_usd"] == 0.002


def test_chat_works_identically_without_manager() -> None:
    app = create_app(registry=_registry(), proxy=FakeProxy())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "say hello"}]},
    )
    assert response.status_code == 200
    assert "X-Budget-Warning" not in response.headers
    assert response.json()["id"] == "chatcmpl-test"
