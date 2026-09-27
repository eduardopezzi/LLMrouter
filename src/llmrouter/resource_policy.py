"""ResourcePolicy contract — PRecog ↔ LLMRouter (M6).

Classificação de campos (F2):
- ``enforced-by-router``: max_context_tokens / max_output_tokens — o router
  recusa (422) valores inválidos e hard-encestra no caminho non-stream.
- ``advisory``: memory budgets, top_k, cache, fallback — o PRecog consome;
  o router valida FORMA (tipos/faixas), não valor semântico.

O schema é versionado (``version``); payload inválido → ``PolicyValidationError``
→ 422 claro no proxy. NUNCA crash, nunca silêncio.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PolicyEnforcement(StrEnum):
    """Como o router trata cada campo."""

    ENFORCED = "enforced-by-router"
    ADVISORY = "advisory"


class ResourcePolicy(BaseModel):
    """Contrato enviado pelo PRecog (header X-Resource-Policy + body versionado)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=r"^\d+$", description="Versão do contrato (ex.: '1').")
    model_class: str = Field(min_length=1, max_length=64)
    reasoning: str = Field(default="standard", max_length=32)

    # enforced-by-router
    max_context_tokens: int = Field(ge=256, le=1_000_000)
    max_output_tokens: int = Field(ge=1, le=100_000)

    # advisory (PRecog consome; router valida forma)
    working_memory_tokens: int = Field(default=0, ge=0, le=200_000)
    episodic_memory_tokens: int = Field(default=0, ge=0, le=200_000)
    semantic_memory_tokens: int = Field(default=0, ge=0, le=200_000)
    procedural_memory_tokens: int = Field(default=0, ge=0, le=200_000)
    knowledge_top_k: int = Field(default=10, ge=1, le=100)

    cache_enabled: bool = True
    fallback_model_class: str | None = Field(default=None, max_length=64)

    @field_validator("max_output_tokens")
    @classmethod
    def _output_fits_context(cls, v: int, info: Any) -> int:
        ctx = info.data.get("max_context_tokens")
        if isinstance(ctx, int) and v > ctx:
            raise ValueError(
                f"max_output_tokens ({v}) não pode exceder max_context_tokens ({ctx})"
            )
        return v

    @property
    def enforced_fields(self) -> dict[str, int]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
        }

    @property
    def advisory_fields(self) -> dict[str, int]:
        return {
            "working_memory_tokens": self.working_memory_tokens,
            "episodic_memory_tokens": self.episodic_memory_tokens,
            "semantic_memory_tokens": self.semantic_memory_tokens,
            "procedural_memory_tokens": self.procedural_memory_tokens,
            "knowledge_top_k": self.knowledge_top_k,
        }


class PolicyValidationError(ValueError):
    """Payload de ResourcePolicy inválido — mapeia para 422 no proxy."""


def parse_resource_policy(payload: str | bytes | dict[str, Any]) -> ResourcePolicy:
    """Parse defensivo do payload (header JSON ou dict).

    Levanta ``PolicyValidationError`` (não ValidationError cru) para o proxy
    traduzir em 422 com mensagem clara — contrato do M6.
    """
    import json

    try:
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else dict(payload)
        return ResourcePolicy.model_validate(data)
    except Exception as exc:  # ValidationError, JSONDecodeError, TypeError
        raise PolicyValidationError(f"invalid resource policy payload: {exc}") from exc


__all__ = [
    "PolicyEnforcement",
    "PolicyValidationError",
    "ResourcePolicy",
    "parse_resource_policy",
]
