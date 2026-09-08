"""Core domain types for the LLM router.

All types are immutable (frozen dataclasses) to enable functional-style
processing and safe sharing across async tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ZAI = "zai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


class Tier(int, Enum):
    """Model capability tiers.

    Tier 1 = simple/cheap models (fast responses, low cost).
    Tier 2 = mid-tier models (balanced cost/quality).
    Tier 3 = high-end models (best quality, higher cost).
    """

    T1 = 1
    T2 = 2
    T3 = 3


class RoutingGrade(str, Enum):
    """Quality assessment of a routing decision."""

    OPTIMAL = "optimal"
    OVERKILL = "overkill"  # Model too powerful/expensive for the task
    UNDERKILL = "underkill"  # Model too weak for the task
    CORRECT = "correct"  # Correct tier but sub-optimal model


class RoutingStrategy(str, Enum):
    """Strategy for selecting models within a tier."""

    COST = "cost"
    QUALITY = "quality"
    BALANCED = "balanced"
    LATENCY = "latency"


class FinishReason(str, Enum):
    """Reason the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Model & registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Immutable description of a registered model.

    Attributes:
        name: The canonical model name (e.g. ``gpt-4o``).
        provider: Which provider hosts this model.
        tier: Capability tier (1=cheap, 3=powerful).
        cost_per_1k_input: Cost in USD per 1,000 input tokens.
        cost_per_1k_output: Cost in USD per 1,000 output tokens.
        max_tokens: Maximum output tokens the model can generate.
        capabilities: Set of capability tags (e.g. ``code``, ``vision``).
        priority: Tie-breaker priority within the same tier (lower = preferred).
        context_window: Maximum input context length in tokens.
        api_base: Optional provider endpoint declared by the model catalog.
        description: Human-facing description from the catalog.
        rollout_percentage: Traffic percentage this model receives during
            a canary/blue-green rollout (0–100). Defaults to ``100`` (full traffic).
            Set to ``0`` to instantly remove from routing without deleting the entry.
        benchmark_scores: Raw public benchmark measurements as ``(name, value)``
            pairs. Percentages may use 0–100; normalized values may use 0–1;
            Codeforces may use its native rating scale.
        benchmark_sources: Validated benchmark URLs associated with this model.
            They are informational provenance, never routing signals by
            themselves.
    """

    name: str
    provider: Provider
    tier: Tier
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    capabilities: frozenset[str] = field(default_factory=frozenset)
    priority: int = 10
    context_window: int = 8192
    api_base: str | None = None
    description: str = ""
    rollout_percentage: float = 100.0
    benchmark_scores: tuple[tuple[str, float], ...] = ()
    benchmark_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate rollout_percentage range."""
        if not 0.0 <= self.rollout_percentage <= 100.0:
            raise ValueError(
                f"rollout_percentage must be between 0 and 100, "
                f"got {self.rollout_percentage} for model '{self.name}'"
            )

    @property
    def cost_ratio(self) -> float:
        """Approximate cost ratio (input + output weighted equally)."""
        return self.cost_per_1k_input + self.cost_per_1k_output

    @property
    def provider_model_name(self) -> str:
        """Model name expected by the upstream provider API."""
        prefixes = {
            Provider.OLLAMA: "ollama/",
            Provider.ZAI: "zhipu/",
            Provider.DEEPSEEK: "deepseek/",
        }
        prefix = prefixes.get(self.provider)
        if prefix and self.name.startswith(prefix):
            return self.name[len(prefix) :]
        return self.name


# ---------------------------------------------------------------------------
# Chat request/response (OpenAI-compatible)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a chat conversation."""

    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    """Normalized chat completion request (OpenAI-compatible subset).

    Extra fields are preserved in ``extra`` for passthrough to providers.
    """

    model: str | None  # None means "router decides"
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    top_p: float = 1.0
    stop: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Concatenate all message contents for scoring."""
        return "\n".join(filter(None, (self._message_text(message) for message in self.messages)))

    def routing_prompt_text(self, max_chars: int = 12_000) -> str:
        """Return bounded, current-intent context for routing decisions.

        The provider still receives :attr:`prompt_text`. Routing must not let an
        old transcript or injected memory dominate the current task classifier.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")

        system_prefix = next(
            (self._message_text(message) for message in self.messages if message.role == "system"),
            "",
        )[: min(2_000, max_chars)]
        recent = [message for message in self.messages if message.role != "system"][-4:]
        recent_text = "\n".join(
            f"{message.role}: {self._message_text(message)}"
            for message in recent
            if self._message_text(message)
        )
        if not recent_text:
            recent_text = system_prefix
            system_prefix = ""
        if system_prefix and recent_text:
            available = max_chars - len(system_prefix) - 1
            recent_text = self._bounded_text(recent_text, available)
            return self._bounded_text(f"{system_prefix}\n{recent_text}", max_chars)
        return self._bounded_text(recent_text, max_chars)

    @staticmethod
    def _bounded_text(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 32:
            return text[:max_chars]
        marker = "\n...[truncated]...\n"
        budget = max_chars - len(marker)
        head = budget // 2
        return f"{text[:head]}{marker}{text[-(budget - head):]}"

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            return "\n".join(
                str(block.get("text") or block.get("content") or "")
                for block in message.content
                if isinstance(block, dict)
                and isinstance(block.get("text") or block.get("content") or "", str)
            )
        return ""


@dataclass(frozen=True)
class Usage:
    """Token usage statistics with explicit cache-reporting provenance."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int | None = None
    cache_status: str = "not_reported"


@dataclass(frozen=True)
class ChatResponse:
    """Normalized chat completion response (OpenAI-compatible)."""

    id: str
    model: str
    choices: list[dict[str, Any]]
    usage: Usage
    finish_reason: FinishReason = FinishReason.STOP
    created: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingConstraints:
    """Constraints applied during model selection.

    Attributes:
        max_cost_per_request: Maximum allowed cost (USD) per request.
        required_capabilities: Capabilities the model must have.
        preferred_tier: If set, only consider models in this tier or higher.
        max_latency_ms: Soft latency target (for future use).
        preferred_provider: Optional provider to prefer for client affinity.
    """

    max_cost_per_request: float | None = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    preferred_tier: Tier | None = None
    max_latency_ms: float | None = None
    preferred_provider: Provider | None = None


@dataclass(frozen=True)
class RoutingDecision:
    """The result of a routing operation.

    Attributes:
        primary: The chosen model.
        fallbacks: Ordered list of fallback models.
        score: Complexity score (0.0–1.0) from the scorer.
        tier: The tier selected by the scorer.
        reason: Human-readable explanation of the decision.
        rollout_sampled: ``"model_name:percentage"`` when the primary was selected
            via rollout filtering (i.e. had ``rollout_percentage < 100``).
            ``None`` when no rollout filtering was applied.
        probe_models: Half-open cooldown models to canary in the background while
            this request is served by an available alternative.
    """

    primary: ModelInfo
    fallbacks: list[ModelInfo]
    score: float
    tier: Tier
    reason: str
    rollout_sampled: str | None = None
    probe_models: tuple[ModelInfo, ...] = ()
