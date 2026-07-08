"""Tests for the Canary/Blue-Green Model Rollout feature (Roadmap #9).

Covers:
- ``ModelInfo.rollout_percentage`` validation
- ``_model_from_mapping`` parsing from YAML
- ``MultiModelRouter._apply_rollout`` hash-based deterministic filtering
- Safety net returns [] when all filtered (rollback instantâneo)
- Routing integration (``route()`` populates ``rollout_sampled`` only on primary)
- ``set_model_rollout_percentage`` YAML persistence with file lock
- ``RolloutConfig`` in Settings and effect on router
"""

from __future__ import annotations

import pytest

from llmrouter.cli_panel import set_model_rollout_percentage
from llmrouter.config import RolloutConfig, Settings
from llmrouter.core.registry import _model_from_mapping, load_model_registry
from llmrouter.core.router import MultiModelRouter
from llmrouter.core.scorer import PromptScorer
from llmrouter.core.types import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Provider,
    RoutingStrategy,
    Tier,
)
from llmrouter.core.registry import ModelRegistry


# ---------------------------------------------------------------------------
# ModelInfo validation
# ---------------------------------------------------------------------------


def _make_model(
    name: str = "test-model",
    rollout_percentage: float = 100.0,
    tier: Tier = Tier.T2,
) -> ModelInfo:
    return ModelInfo(
        name=name,
        provider=Provider.OLLAMA,
        tier=tier,
        rollout_percentage=rollout_percentage,
    )


class TestModelInfoRolloutValidation:
    """Tests for ``ModelInfo.rollout_percentage`` field."""

    def test_default_rollout_is_100(self):
        model = ModelInfo(name="m", provider=Provider.OLLAMA, tier=Tier.T1)
        assert model.rollout_percentage == 100.0

    def test_valid_rollout_values(self):
        for pct in [0.0, 0.1, 5.0, 50.0, 99.9, 100.0]:
            model = _make_model(rollout_percentage=pct)
            assert model.rollout_percentage == pct

    def test_negative_rollout_raises(self):
        with pytest.raises(ValueError, match="rollout_percentage"):
            _make_model(rollout_percentage=-1.0)

    def test_over_100_rollout_raises(self):
        with pytest.raises(ValueError, match="rollout_percentage"):
            _make_model(rollout_percentage=100.1)


# ---------------------------------------------------------------------------
# Registry loader parsing
# ---------------------------------------------------------------------------


class TestRegistryLoaderRollout:
    """Tests for YAML rollout_percentage parsing."""

    def test_parses_rollout_percentage(self):
        item = {
            "name": "test/model",
            "provider": "ollama",
            "rollout_percentage": 25,
        }
        model = _model_from_mapping(item)
        assert model.rollout_percentage == 25.0

    def test_defaults_to_100_when_absent(self):
        item = {"name": "test/model", "provider": "ollama"}
        model = _model_from_mapping(item)
        assert model.rollout_percentage == 100.0

    def test_clamps_above_100_now_raises(self):
        """Previously clamped to 100.0; now ModelInfo rejects >100."""
        item = {
            "name": "test/model",
            "provider": "ollama",
            "rollout_percentage": 150,
        }
        with pytest.raises(ValueError, match="rollout_percentage"):
            _model_from_mapping(item)

    def test_clamps_below_zero_now_raises(self):
        """Previously clamped to 0.0; now ModelInfo rejects <0."""
        item = {
            "name": "test/model",
            "provider": "ollama",
            "rollout_percentage": -5,
        }
        with pytest.raises(ValueError, match="rollout_percentage"):
            _model_from_mapping(item)


# ---------------------------------------------------------------------------
# _apply_rollout unit tests
# ---------------------------------------------------------------------------


def _make_request(prompt: str = "test prompt") -> ChatRequest:
    return ChatRequest(model=None, messages=[ChatMessage(role="user", content=prompt)])


class TestApplyRollout:
    """Unit tests for ``MultiModelRouter._apply_rollout``."""

    def _make_router(self, models: list[ModelInfo]) -> MultiModelRouter:
        registry = ModelRegistry(models=tuple(models))
        return MultiModelRouter(
            registry=registry,
            scorer=PromptScorer(),
            strategy=RoutingStrategy.COST,
        )

    def test_all_100_returns_unchanged(self):
        models = [_make_model("a"), _make_model("b")]
        router = self._make_router(models)
        request = _make_request("prompt")

        filtered = router._apply_rollout(models, request)

        assert filtered is models  # same list object, no copy

    def test_rollout_zero_filters_model(self):
        """Model with rollout=0 should never appear in filtered list."""
        model_zero = _make_model("zero", rollout_percentage=0.0)
        model_full = _make_model("full", rollout_percentage=100.0)
        router = self._make_router([model_zero, model_full])
        request = _make_request("prompt")

        filtered = router._apply_rollout([model_zero, model_full], request)

        assert model_zero not in filtered
        assert model_full in filtered

    def test_deterministic_same_prompt_same_result(self):
        """Same prompt + same model must produce consistent inclusion/exclusion."""
        model_50 = _make_model("canary", rollout_percentage=50.0)
        router = self._make_router([model_50])
        request = _make_request("identical prompt text")

        r1 = router._apply_rollout([model_50], request)
        r2 = router._apply_rollout([model_50], request)
        r3 = router._apply_rollout([model_50], request)

        # The result must be deterministic (all same)
        assert r1 == r2 == r3

    def test_safety_net_returns_empty_when_all_filtered(self):
        """If all models are rollout=0, safety net returns empty list."""
        model_zero = _make_model("zero", rollout_percentage=0.0)
        model_zero2 = _make_model("zero2", rollout_percentage=0.0)
        router = self._make_router([model_zero, model_zero2])
        request = _make_request("prompt")

        filtered = router._apply_rollout([model_zero, model_zero2], request)

        assert filtered == []
        # Downstream fallback should activate (no safety net bypass)

    def test_partial_rollout_included(self):
        """When a model with rollout < 100 survives, it appears in filtered list."""
        model_50 = _make_model("canary", rollout_percentage=50.0)
        model_full = _make_model("stable", rollout_percentage=100.0)
        router = self._make_router([model_50, model_full])
        request = _make_request("prompt")

        for i in range(200):
            req = _make_request(f"test prompt number {i}")
            filtered = router._apply_rollout([model_50, model_full], req)
            if model_50 in filtered:
                return

        pytest.fail("Canary model never appeared in 200 attempts with 50% rollout")

    def test_statistical_distribution(self):
        """Simulate many requests and verify rollout percentage is roughly correct."""
        model_30 = _make_model("canary30", rollout_percentage=30.0)
        model_full = _make_model("stable", rollout_percentage=100.0)
        router = self._make_router([model_30, model_full])

        canary_count = 0
        total = 2000

        for i in range(total):
            req = _make_request(f"distribution test prompt {i}")
            filtered = router._apply_rollout([model_30, model_full], req)
            if model_30 in filtered:
                canary_count += 1

        ratio = canary_count / total
        # 30% rollout should produce roughly 25-35% (with some tolerance)
        assert 0.20 < ratio < 0.40, f"Canary ratio {ratio:.2%} outside expected range for 30% rollout"

    def test_rollout_disabled_returns_all(self):
        """When RolloutConfig.enabled=False, all models pass through."""
        model_0 = _make_model("zero", rollout_percentage=0.0)
        model_50 = _make_model("canary", rollout_percentage=50.0)
        router = MultiModelRouter(
            registry=ModelRegistry(models=(model_0, model_50)),
            scorer=PromptScorer(),
            strategy=RoutingStrategy.COST,
            rollout_config=RolloutConfig(enabled=False),
        )
        request = _make_request("prompt")
        filtered = router._apply_rollout([model_0, model_50], request)

        # All models pass through when rollout is disabled
        assert model_0 in filtered
        assert model_50 in filtered


# ---------------------------------------------------------------------------
# Routing integration
# ---------------------------------------------------------------------------


class TestRoutingIntegration:
    """Tests for rollout integration in ``MultiModelRouter.route()``."""

    @pytest.mark.asyncio
    async def test_route_with_rollout_never_sets_sampled_when_all_100(self):
        """Route should never set rollout_sampled when all models are 100%."""
        models = [_make_model("a", tier=Tier.T1), _make_model("b", tier=Tier.T1)]
        registry = ModelRegistry(models=tuple(models))
        router = MultiModelRouter(
            registry=registry,
            scorer=PromptScorer(),
            strategy=RoutingStrategy.COST,
        )

        request = _make_request("simple text")
        decision = await router.route(request)

        assert decision.rollout_sampled is None

    @pytest.mark.asyncio
    async def test_route_sets_rollout_sampled_when_primary_is_canary(self):
        """rollout_sampled should be set only when the *primary* model has rollout < 100."""
        # Canary has higher priority (lower number) so the strategy picks it
        # over stable when both survive the rollout filter.
        canary = ModelInfo(
            name="canary", provider=Provider.OLLAMA, tier=Tier.T1,
            priority=1, rollout_percentage=50.0,
        )
        stable = ModelInfo(
            name="stable", provider=Provider.OLLAMA, tier=Tier.T1,
            priority=10, rollout_percentage=100.0,
        )

        registry = ModelRegistry(models=(stable, canary))
        router = MultiModelRouter(
            registry=registry,
            scorer=PromptScorer(),
            strategy=RoutingStrategy.QUALITY,
        )

        found_canary_as_primary = False
        for i in range(500):
            request = _make_request(f"canary integration test {i}")
            decision = await router.route(request)
            if decision.rollout_sampled is not None:
                assert "canary" in decision.rollout_sampled
                assert "50" in decision.rollout_sampled
                # Confirm the primary is indeed the canary
                assert decision.primary.rollout_percentage < 100.0
                assert decision.primary.name == "canary"
                found_canary_as_primary = True
                break

        assert found_canary_as_primary, "Canary never selected as primary in 500 attempts"


# ---------------------------------------------------------------------------
# CLI panel: set_model_rollout_percentage
# ---------------------------------------------------------------------------


class TestSetModelRolloutPercentage:
    """Tests for ``set_model_rollout_percentage`` YAML persistence."""

    def test_set_rollout_on_existing_field(self, tmp_path):
        yaml_content = """\
models:
  - name: "test/model-a"
    provider: "ollama"
    priority: 1
    rollout_percentage: 50
"""
        path = tmp_path / "models.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        set_model_rollout_percentage(path, "test/model-a", 25.0)

        registry = load_model_registry(path)
        model = registry.get("test/model-a")
        assert model is not None
        assert model.rollout_percentage == 25.0

    def test_set_rollout_inserts_new_field(self, tmp_path):
        yaml_content = """\
models:
  - name: "test/model-a"
    provider: "ollama"
    priority: 1
"""
        path = tmp_path / "models.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        set_model_rollout_percentage(path, "test/model-a", 10.0)

        registry = load_model_registry(path)
        model = registry.get("test/model-a")
        assert model is not None
        assert model.rollout_percentage == 10.0

    def test_set_rollout_invalid_model_raises(self, tmp_path):
        yaml_content = """\
models:
  - name: "test/model-a"
    provider: "ollama"
"""
        path = tmp_path / "models.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        with pytest.raises(ValueError, match="model not found"):
            set_model_rollout_percentage(path, "nonexistent/model", 50.0)

    def test_set_rollout_invalid_percentage_raises(self, tmp_path):
        yaml_content = """\
models:
  - name: "test/model-a"
    provider: "ollama"
"""
        path = tmp_path / "models.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        with pytest.raises(ValueError, match="rollout percentage"):
            set_model_rollout_percentage(path, "test/model-a", 150.0)

    def test_rollback_to_zero(self, tmp_path):
        """Setting rollout to 0 should effectively disable the model."""
        yaml_content = """\
models:
  - name: "test/model-a"
    provider: "ollama"
    rollout_percentage: 100
"""
        path = tmp_path / "models.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        set_model_rollout_percentage(path, "test/model-a", 0.0)

        registry = load_model_registry(path)
        model = registry.get("test/model-a")
        assert model is not None
        assert model.rollout_percentage == 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestRolloutConfig:
    """Tests for ``RolloutConfig`` in Settings."""

    def test_rollout_config_defaults(self):
        config = RolloutConfig()
        assert config.enabled is True
        assert config.deterministic is True
        assert config.critical_threshold_pct == 5.0
        assert config.auto_promote is False
        assert config.auto_rollback_error_rate == 20.0

    def test_settings_has_rollout_config(self):
        settings = Settings()
        assert hasattr(settings, "rollout")
        assert isinstance(settings.rollout, RolloutConfig)
        assert settings.rollout.enabled is True

    def test_rollout_config_env_override(self, monkeypatch):
        monkeypatch.setenv("LLMROUTER_ROLLOUT__ENABLED", "false")
        monkeypatch.setenv("LLMROUTER_ROLLOUT__DETERMINISTIC", "false")
        settings = Settings()
        assert settings.rollout.enabled is False
        assert settings.rollout.deterministic is False

    def test_rollout_config_injected_into_router(self):
        """Verify RolloutConfig.enabled=False disables filtering."""
        model_0 = _make_model("zero", rollout_percentage=0.0)
        router = MultiModelRouter(
            registry=ModelRegistry(models=(model_0,)),
            scorer=PromptScorer(),
            strategy=RoutingStrategy.COST,
            rollout_config=RolloutConfig(enabled=False),
        )
        request = _make_request("prompt")
        filtered = router._apply_rollout([model_0], request)
        assert len(filtered) == 1
        assert filtered[0].name == "zero"