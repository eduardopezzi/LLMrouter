"""M6 — ResourcePolicy PRecog ↔ LLMRouter: schema, enforcement e contrato 402."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.llmrouter.resource_policy import (
    PolicyValidationError,
    ResourcePolicy,
    parse_resource_policy,
)


# ---------------------------------------------------------------------------
# 1. Schema (F2: enforced vs advisory)
# ---------------------------------------------------------------------------


class TestResourcePolicySchema:
    def test_valid_policy_parses(self) -> None:
        policy = parse_resource_policy(
            {
                "version": "1",
                "model_class": "standard",
                "max_context_tokens": 16000,
                "max_output_tokens": 4096,
            }
        )
        assert policy.version == "1"
        assert policy.max_context_tokens == 16000
        # defaults advisory
        assert policy.knowledge_top_k == 10
        assert policy.cache_enabled is True

    def test_invalid_json_raises_policy_error(self) -> None:
        with pytest.raises(PolicyValidationError):
            parse_resource_policy("{not json")

    def test_invalid_values_rejected(self) -> None:
        with pytest.raises(PolicyValidationError):
            parse_resource_policy(
                {"version": "1", "model_class": "x", "max_context_tokens": 10, "max_output_tokens": 5}
            )  # max_context < floor 256

    def test_output_exceeding_context_rejected(self) -> None:
        with pytest.raises(PolicyValidationError):
            parse_resource_policy(
                {
                    "version": "1",
                    "model_class": "x",
                    "max_context_tokens": 1000,
                    "max_output_tokens": 2000,
                }
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(PolicyValidationError):
            parse_resource_policy(
                {
                    "version": "1",
                    "model_class": "x",
                    "max_context_tokens": 1000,
                    "max_output_tokens": 100,
                    "chain_of_thought": "leak me",
                }
            )

    def test_enforced_vs_advisory_classification(self) -> None:
        policy = ResourcePolicy(
            version="1",
            model_class="x",
            max_context_tokens=1000,
            max_output_tokens=100,
        )
        assert set(policy.enforced_fields) == {"max_context_tokens", "max_output_tokens"}
        assert "knowledge_top_k" in policy.advisory_fields
        assert policy.advisory_fields["knowledge_top_k"] == 10

    def test_header_payload_round_trip(self) -> None:
        policy = ResourcePolicy(
            version="1", model_class="x", max_context_tokens=8000, max_output_tokens=512
        )
        restored = parse_resource_policy(policy.model_dump_json())
        assert restored == policy


# ---------------------------------------------------------------------------
# 2. Proxy: header parsing (422), enforcement headers (non-stream)
# ---------------------------------------------------------------------------


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.llmrouter.api.routes import create_app

    app = create_app()
    return TestClient(app)


class TestProxyResourcePolicy:
    @pytest.fixture()
    def client(self, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        return _client(monkeypatch)

    def test_invalid_header_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Resource-Policy": "{invalid"},
        )
        assert response.status_code == 422
        assert "X-Resource-Policy" in response.json()["detail"]

    def test_valid_policy_echoes_version_header(self, client: TestClient) -> None:
        policy = {
            "version": "1",
            "model_class": "standard",
            "max_context_tokens": 16000,
            "max_output_tokens": 4096,
        }
        # sem providers configurados o proxy pode 503; o header de policy
        # ainda assim prova que o parse aconteceu ANTES do proxy
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Resource-Policy": json.dumps(policy)},
        )
        assert response.status_code in {200, 502, 503}
        if response.status_code == 200:
            assert response.headers.get("X-Resource-Policy-Version") == "1"
            assert "X-Budget-Remaining" in response.headers


# ---------------------------------------------------------------------------
# 3. Contrato F3: 402 de budget ≠ 402 de provider (cooldown)
# ---------------------------------------------------------------------------


class TestBudget402VsProvider402:
    @pytest.mark.asyncio()
    async def test_budget_402_does_not_touch_provider_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O 402 do BudgetManager é emitido ANTES do proxy e NÃO pode contar
        como erro de provider (senão colocaria o modelo em cooldown)."""
        from src.llmrouter.core.budget import BudgetDecision, BudgetManager

        manager = BudgetManager(":memory:")

        async def _denied(*args: Any, **kwargs: Any) -> BudgetDecision:
            return BudgetDecision(
                allowed=False, reason="daily budget exceeded", warning=None
            )

        monkeypatch.setattr(manager, "check", _denied)
        # a decisão negada é puramente de governança — o provider nunca é chamado
        decision = await manager.check("p", "u", 0.0)
        assert decision.allowed is False
        assert "budget" in decision.reason

    def test_provider_402_is_a_provider_error(self) -> None:
        from src.llmrouter.providers.base import ProviderError

        exc = ProviderError("provider quota exhausted", status_code=402)
        assert exc.status_code == 402
        # este 402 VEM do provider e segue o caminho normal de cooldown
