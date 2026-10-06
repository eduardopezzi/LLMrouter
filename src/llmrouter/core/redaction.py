"""Redação determinística de prompts (LGPD/PII + segredos).

Estratégia: regex por categoria, sem LLM, sem dependência externa.
Cada match em resultado vira um placeholder curto do tipo ``[CAT]``
(ex.: ``[EMAIL]``, ``[DOC_ID]``, ``[PHONE]``, ``[IP]``, ``[PAYMENT]``,
``[REDACTED:TOKEN]``) — exceto em ``hash_mode=True`` onde o valor é
substituído por um SHA-256 truncado de 12 chars (preserva capacidade de
agregação/contagem sem expor o valor).

Camadas (cada nível inclui as anteriores):

    NONE       : nada
    PII        : EMAIL, PHONE, CPF/CNPJ (validado por dígito verificador),
                   PAYMENT (cartão Luhn), IPV4
    SECRETS    : PII + tokens (ghp_, sk-, ragflow-, JWT, bearer genérico)
    STRICT     : SECRETS + paths absolutos + pares password= / token=

Default recomendado: ``SECRETS`` para publicação externa
(``/v1/llmrouter/feedback`` → PRecog).
"""

from __future__ import annotations

import enum
import hashlib
import re
from collections.abc import Callable


class PromptRedactionLevel(str, enum.Enum):
    NONE = "none"
    PII = "pii"
    SECRETS = "secrets"
    STRICT = "strict"


# Helper ---------------------------------------------------------------

def _hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


# Patterns -------------------------------------------------------------
# Cada tupla: (nome, regex, placeholder). A ordem importa: tokens vêm
# antes de EMAIL porque a regex de EMAIL não casa o prefixo `ghp_`.

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+")

# Telefones BR simples (com ou sem DDI). Casos ambíguos (números curtos)
# são deixados passar — o custo de falso positivo num telefone de 7 dígitos
# é maior que o ganho de redação.
_PHONE_RE = re.compile(r"\+?\d{1,3}[\s.-]?\(?\d{2}\)?[\s.-]?\d{4,5}[\s.-]?\d{4}")

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# CPF/CNPJ. O lookahead/budget evita capturar partes maiores (ex. CNPJ
# dentro de telefone), e o dígito verificador é checado separadamente
# para evitar falsos positivos em números aleatórios com mesmo formato.
_CPF_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_CNPJ_RE = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")

# Cartão de crédito (13-19 dígitos, separadores opcionais). Validado por
# Luhn no pós-processamento para reduzir falsos positivos.
_CC_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = (len(digits) - 2) % 2
    for i, d in enumerate(digits[:-1]):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return (checksum + digits[-1]) % 10 == 0


def _cpf_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False

    def dv(d: list[int]) -> int:
        s = sum(d[i] * (len(d) + 1 - i) for i in range(len(d)))
        r = s % 11
        return 0 if r < 2 else 11 - r

    return (
        dv(digits[:9]) == digits[9]
        and dv(digits[:10]) == digits[10]
    )


def _cnpj_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if len(digits) != 14 or len(set(digits)) == 1:
        return False

    def dv(d: list[int], weights: list[int]) -> int:
        s = sum(d[i] * weights[i] for i in range(len(d)))
        r = s % 11
        return 0 if r < 2 else 11 - r

    return (
        dv(digits[:12], [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]) == digits[12]
        and dv(digits[:13], [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]) == digits[13]
    )


# Tokens genéricos: prefixos comuns (ghp_, sk-, sk-antig…),
# ragflow-, ghp_, plus Authorization: Bearer xxxx
_TOKEN_PREFIX_RE = re.compile(
    r"\b(?:ghp_|sk-|sk-antig|pk-|rk-|ragflow-|llmrouter-)[A-Za-z0-9._-]{8,}\b"
)
# JWT: 3 segmentos base64url separados por '.', segmento 1 com 8+ chars.
_JWT_RE = re.compile(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{12,}")

# Pares password= / token= / api_key= (heurística — valor até
# whitespace ou fim de string/linha).
_KV_RE = re.compile(
    r"(?i)\b(password|passwd|token|api_key|apikey|secret)\s*=\s*[^\s,;&\"']+"
)

# Paths absolutos (Unix + Windows). Heurística: começa com / ou C:/
_ABS_PATH_RE = re.compile(r"(?:/[A-Za-z0-9_.-]+){2,}\S*|(?:[A-Z]:[\\/][^\s,'\"]+)")


def _replace(
    text: str,
    pattern: re.Pattern[str],
    tag: str,
    *,
    hash_mode: bool,
    validator: Callable[[str], bool] | None = None,
) -> str:
    def _sub(m: re.Match[str]) -> str:
        value = m.group(0)
        if validator and not validator(value):
            return value
        return f"[{_hash(value)}]" if hash_mode else f"[{tag}]"

    return pattern.sub(_sub, text)


# Public ---------------------------------------------------------------

def redact(
    text: str,
    *,
    level: PromptRedactionLevel = PromptRedactionLevel.SECRETS,
    hash_mode: bool = False,
) -> str:
    """Redact PII and secrets in ``text`` according to ``level``.

    ``hash_mode=True`` substitui cada match por SHA-256 truncado
    (preserva contagem sem expor valor). Útil p/ agregações estatísticas
    sem expor o valor.
    """
    if level == PromptRedactionLevel.NONE or not text:
        return text

    # Tokens first (mais específicos) — antes de EMAIL.
    text = _replace(text, _TOKEN_PREFIX_RE, "REDACTED:TOKEN", hash_mode=hash_mode)
    text = _replace(text, _JWT_RE, "REDACTED:TOKEN", hash_mode=hash_mode)
    text = _replace(text, _BEARER_RE, "REDACTED:TOKEN", hash_mode=hash_mode)
    text = _replace(text, _KV_RE, "REDACTED:KV", hash_mode=hash_mode)

    # PII (sempre presente de PII em diante)
    text = _replace(text, _EMAIL_RE, "EMAIL", hash_mode=hash_mode)
    text = _replace(text, _PHONE_RE, "PHONE", hash_mode=hash_mode)
    text = _replace(text, _CPF_RE, "DOC_ID", hash_mode=hash_mode, validator=_cpf_ok)
    text = _replace(text, _CNPJ_RE, "DOC_ID", hash_mode=hash_mode, validator=_cnpj_ok)
    text = _replace(text, _CC_RE, "PAYMENT", hash_mode=hash_mode, validator=_luhn_ok)
    text = _replace(text, _IPV4_RE, "IP", hash_mode=hash_mode)

    if level == PromptRedactionLevel.STRICT:
        text = _replace(text, _ABS_PATH_RE, "PATH", hash_mode=hash_mode)

    return text


__all__ = ["PromptRedactionLevel", "redact"]
