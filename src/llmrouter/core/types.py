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
    NVIDIA = "nvidia"


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

    @property
    def cost_ratio(self) -> float:
        """Approximate cost ratio (input + output weighted equally)."""
        return self.cost_per_1k_input + self.cost_per_1k_output

    @property
    def provider_model_name(self) -> str:
        """Model name expected by the upstream provider API."""
        prefixes = {
            Provider.OLLAMA: "ollama/",
            Provider.NVIDIA: "nvidia_nim/",
            Provider.ZAI: "zhipu/",
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
    content: str
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
        return "\n".join(m.content for m in self.messages if m.content)


@dataclass(frozen=True)
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    """Normalized chat completion response (OpenAI-compatible)."""

    id: str
    model: str
    choices: list[dict[str, Any]]
    usage: Usage
    finish_reason: FinishReason = FinishReason.STOP
    created: int = 0


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
    """

    max_cost_per_request: float | None = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    preferred_tier: Tier | None = None
    max_latency_ms: float | None = None


@dataclass(frozen=True)
class RoutingDecision:
    """The result of a routing operation.

    Attributes:
        primary: The chosen model.
        fallbacks: Ordered list of fallback models.
        score: Complexity score (0.0–1.0) from the scorer.
        tier: The tier selected by the scorer.
        reason: Human-readable explanation of the decision.
    """

    primary: ModelInfo
    fallbacks: list[ModelInfo]
    score: float
    tier: Tier
    reason: str
