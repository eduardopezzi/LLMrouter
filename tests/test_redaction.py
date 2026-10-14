"""Testes de redação de prompts (TDD — RED).

Contrato do módulo core/redaction.py: substituição determinística de
PII / segredos / dados internos antes da persistência das ObservationCollector
(e da propagação p/ publisher externo).

Estratégia: regex + hashing opcional, sem LLM, sem dependência externa.
"""

from llmrouter.core.redaction import PromptRedactionLevel, redact


def test_secrets_redacts_bearer_and_ghp():
    out = redact(
        "Use ghp_abCD1234567890efgh para clonar e ragflow-XxYyZz123456",
        level=PromptRedactionLevel.SECRETS,
    )
    assert "ghp_ab" not in out
    assert "ragflow-Xx" not in out
    assert "[REDACTED:TOKEN]" in out


def test_pii_redacts_cpf():
    out = redact(
        "Meu CPF é 529.982.247-25 e o telefone +55 11 99999-8888",
        level=PromptRedactionLevel.PII,
    )
    assert "529.982.247-25" not in out
    assert "99999-8888" not in out
    assert "[DOC_ID]" in out
    assert "[PHONE]" in out


def test_email_redacted():
    out = redact(
        "Envie para eduardo@vielitech.com.br",
        level=PromptRedactionLevel.PII,
    )
    assert "eduardo@vielitech.com.br" not in out
    assert "[EMAIL]" in out


def test_jwt_redacted():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkifQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    out = redact(f"Token: {jwt}", level=PromptRedactionLevel.SECRETS)
    assert jwt not in out
    # JWT vira um placeholder genérico de token (mesma tag dos secrets)
    assert "[REDACTED:TOKEN]" in out


def test_password_assignment_redacted():
    out = redact(
        "password=minhasenha123 e token=xyz789",
        level=PromptRedactionLevel.SECRETS,
    )
    assert "minhasenha123" not in out
    assert "xyz789" not in out
    assert "[REDACTED" in out


def test_hash_mode_deterministic_and_no_leak():
    a = redact(
        "E-mail: user@empresa.com.br",
        level=PromptRedactionLevel.PII,
        hash_mode=True,
    )
    b = redact(
        "E-mail: user@empresa.com.br",
        level=PromptRedactionLevel.PII,
        hash_mode=True,
    )
    assert a == b
    assert "user@empresa.com.br" not in a
    # Hash mode NÃO bota o placeholder; é um sha curto hexadecimal/alphabet
    assert "[EMAIL]" not in a or len(a) > len("E-mail: [EMAIL]")


def test_none_keeps_everything():
    text = "ghp_12345 cpf 529.982.247-25 user@x.com"
    out = redact(text, level=PromptRedactionLevel.NONE)
    assert text == out


def test_integration_collector_flush_redacts(tmp_path):
    """ObservationCollector.flush() persiste a versão redigida."""
    import asyncio

    from llmrouter.evaluator.collector import ObservationCollector
    from llmrouter.evaluator.types import RoutingObservation

    db = str(tmp_path / "obs.db")
    collector = ObservationCollector(
        db_path=db,
        redaction_level=PromptRedactionLevel.SECRETS,
    )
    collector.record(
        RoutingObservation(
            prompt="prompt com ghp_secrettoken123456789 dentro",
            chosen_model="m1",
            response="ok",
            latency_ms=1.0,
        )
    )
    asyncio.run(collector.flush())

    import sqlite3
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT prompt FROM observations").fetchone()
    assert row is not None
    persisted = row[0]
    assert "ghp_secrettoken" not in persisted
    assert "[REDACTED:TOKEN]" in persisted
