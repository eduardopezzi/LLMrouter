"""P-CHR hit-log tests: persisted per-hit audit log for the semantic cache.

Covers the E1.3 deliverable (ROADMAP):
- every semantic hit appends one row to ``semantic_cache_hit_log``;
- writes are best-effort (never break the served response);
- ``stats()`` derives ``pchr_*`` counters from the log (survives reopen);
- ``verify_pending`` audits pending rows via an injectable judge;
- ``hit_log_enabled=False`` opts out entirely;
- new ``SemanticCacheConfig`` knobs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmrouter.config import SemanticCacheConfig, Settings
from llmrouter.core.semantic_cache import OllamaJudge, SemanticCache
from llmrouter.core.types import ChatResponse, Tier, Usage

_DIM = 64

_PROMPT_A = "What is the capital of France?"
_PROMPT_A_VARIANT = "The capital of France is what?"
_PROMPT_B = "quantum chromodynamics lattice gauge theory"


def _bag_of_words(text: str) -> list[float]:
    vector = [0.0] * _DIM
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()  # noqa: S324
        vector[int(digest, 16) % _DIM] += 1.0
    return vector


class FakeHashEmbedder:
    """Async embedder: same word multiset -> identical vector."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_bag_of_words(text) for text in texts]


def _response(content: str = "Paris") -> ChatResponse:
    return ChatResponse(
        id="resp-1",
        model="gpt-4o",
        choices=[{"message": {"role": "assistant", "content": content}}],
        usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        latency_ms=50.0,
    )


async def _store_default(
    cache: SemanticCache,
    prompt: str = _PROMPT_A,
    **overrides: Any,
) -> bool:
    params: dict[str, Any] = {
        "model": "gpt-4o",
        "tier": Tier.T2,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 100,
    }
    params.update(overrides)
    return await cache.store(_response(), prompt, **params)


async def _hit_lookup(cache: SemanticCache, prompt: str = _PROMPT_A_VARIANT) -> Any:
    """Perform a lookup that is guaranteed to hit (same word multiset)."""
    return await cache.lookup_prompt(
        prompt,
        model="gpt-4o",
        tier=Tier.T2,
        temperature=0.2,
        top_p=0.9,
        max_tokens=100,
    )


def _unit(cosine: float) -> list[float]:
    """Unit vector [cos, sin, 0] whose cosine with [1, 0, 0] is ``cosine``."""
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0]


def _fetch_log_rows(db_path: str) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return list(
            conn.execute("SELECT * FROM semantic_cache_hit_log ORDER BY id").fetchall()
        )


def _count_rows(db_path: str, where: str = "1=1") -> int:
    with sqlite3.connect(db_path) as conn:
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM semantic_cache_hit_log WHERE {where}"  # noqa: S608
            ).fetchone()[0]
        )


class QueueJudge:
    """Sync duck-typed judge consuming a planned verdict queue."""

    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str | None, str | None]] = []

    def __call__(
        self, prompt_text: str | None, response_text: str | None
    ) -> tuple[bool, float | None, str | None]:
        self.calls.append((prompt_text, response_text))
        verdict = self.verdicts.pop(0)
        return verdict, 0.9 if verdict else 0.1, "ok" if verdict else "off"


# ---------------------------------------------------------------------------
# (a) hit writes one row with prompt/similarity/threshold/response
# ---------------------------------------------------------------------------


class TestHitLogWrite:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "hit_log.db")

    @pytest.mark.asyncio
    async def test_hit_writes_row_with_full_payload(self, db_path: str) -> None:
        cache = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), threshold=0.90, hit_log_enabled=True
        )
        await _store_default(cache, _PROMPT_A)

        hit = await _hit_lookup(cache)

        assert hit is not None
        rows = _fetch_log_rows(db_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["prompt_text"] == _PROMPT_A_VARIANT
        assert row["response_text"] == "Paris"
        assert row["model"] == "gpt-4o"
        assert row["tier"] == int(Tier.T2)
        assert row["temperature"] == pytest.approx(0.2)
        assert row["top_p"] == pytest.approx(0.9)
        assert row["max_tokens"] == 100
        assert row["similarity"] == pytest.approx(1.0)
        assert row["threshold"] == pytest.approx(0.90)
        assert row["verified"] == 0
        assert row["verify_ts"] is None
        assert row["verify_method"] is None
        assert row["verify_score"] is None
        assert row["verify_note"] is None

    @pytest.mark.asyncio
    async def test_row_key_matches_served_entry_key(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        rows = _fetch_log_rows(db_path)
        assert len(rows) == 1
        with sqlite3.connect(db_path) as conn:
            entry_key = conn.execute(
                "SELECT key FROM semantic_cache_entries"
            ).fetchone()[0]
        assert rows[0]["key"] == entry_key

    @pytest.mark.asyncio
    async def test_miss_writes_no_row(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)

        result = await cache.lookup_prompt(
            _PROMPT_B,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert result is None
        assert _fetch_log_rows(db_path) == []

    @pytest.mark.asyncio
    async def test_raw_embedding_lookup_logs_row_with_null_prompt(
        self, db_path: str
    ) -> None:
        cache = SemanticCache(db_path, hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A, embedding=[1.0, 0.0, 0.0])

        hit = await cache.lookup(
            [1.0, 0.0, 0.0],
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )

        assert hit is not None
        rows = _fetch_log_rows(db_path)
        assert len(rows) == 1
        assert rows[0]["prompt_text"] is None
        assert rows[0]["prompt_hash"] is None
        assert rows[0]["response_text"] == "Paris"

    @pytest.mark.asyncio
    async def test_two_hits_log_two_rows(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)

        assert await _hit_lookup(cache, _PROMPT_A_VARIANT) is not None
        assert await _hit_lookup(cache, "France capital what the of is?") is not None

        assert len(_fetch_log_rows(db_path)) == 2


# ---------------------------------------------------------------------------
# (b) write failures never break the served response
# ---------------------------------------------------------------------------


class TestHitLogFaultTolerance:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "hit_log_fault.db")

    @pytest.mark.asyncio
    async def test_write_failure_returns_response_anyway(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)

        # Force every hit-log INSERT to fail at the SQLite level.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TRIGGER fail_pchr BEFORE INSERT ON semantic_cache_hit_log
                BEGIN SELECT RAISE(ABORT, 'hit log denied'); END
                """
            )

        hit = await _hit_lookup(cache)

        assert hit is not None
        assert hit.choices[0]["message"]["content"] == "Paris"
        assert hit.usage.cache_status == "semantic_hit"
        assert cache.stats()["semantic_hits"] == 1
        assert _count_rows(db_path) == 0

    @pytest.mark.asyncio
    async def test_missing_table_does_not_crash_stats(self, db_path: str) -> None:
        cache = SemanticCache(db_path, hit_log_enabled=True)

        stats = cache.stats()

        assert stats["pchr_pending"] == 0
        assert stats["pchr_verified_ok"] == 0
        assert stats["pchr_verified_mismatch"] == 0
        assert stats["pchr_precision"] is None
        assert stats["pchr_last_verified_ts"] is None


# ---------------------------------------------------------------------------
# (c) stats() derives pchr_* from the log; survives DB reopen
# ---------------------------------------------------------------------------


class TestStatsFromLog:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "hit_log_stats.db")

    @pytest.mark.asyncio
    async def test_stats_derives_pchr_counters_from_log(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        stats = cache.stats()
        assert stats["semantic_hits"] == 1
        assert stats["semantic_misses"] == 0
        assert stats["semantic_unavailable"] == 0
        assert stats["pchr_pending"] == 1
        assert stats["pchr_verified_ok"] == 0
        assert stats["pchr_verified_mismatch"] == 0
        # No verification yet -> precision omitted (None).
        assert stats["pchr_precision"] is None
        assert stats["pchr_last_verified_ts"] is None

    @pytest.mark.asyncio
    async def test_stats_survives_reopen(self, db_path: str) -> None:
        first = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(first, _PROMPT_A)
        await _hit_lookup(first)

        reopened = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True
        )
        stats = reopened.stats()
        # In-memory hit counter restarts (compat), log-derived ones persist.
        assert stats["semantic_hits"] == 0
        assert stats["pchr_pending"] == 1

    @pytest.mark.asyncio
    async def test_stats_reflects_verdicts_and_precision(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)  # row 1
        await _hit_lookup(cache, "France capital what the of is?")  # row 2

        judge = QueueJudge([True, False])
        result = await cache.verify_pending(10, judge)
        assert result["checked"] == 2

        stats = cache.stats()
        assert stats["pchr_pending"] == 0
        assert stats["pchr_verified_ok"] == 1
        assert stats["pchr_verified_mismatch"] == 1
        assert stats["pchr_precision"] == pytest.approx(0.5)
        assert stats["pchr_last_verified_ts"] is not None
        assert stats["pchr_last_verified_ts"] > 0


# ---------------------------------------------------------------------------
# (d)(e)(g)(h) verify_pending
# ---------------------------------------------------------------------------


class TestVerifyPending:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "hit_log_verify.db")

    @pytest.mark.asyncio
    async def test_verify_marks_ok_and_mismatch(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)  # row 1 (most recent)
        await _hit_lookup(cache, "France capital what the of is?")  # row 2

        result = await cache.verify_pending(10, QueueJudge([True, False]))

        assert result["checked"] == 2
        assert result["ok"] == 1
        assert result["mismatch"] == 1
        assert result["error"] == 0

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM semantic_cache_hit_log ORDER BY id DESC"
            ).fetchall()
        assert rows[0]["verified"] == 1
        assert rows[0]["verify_ts"] is not None
        assert rows[0]["verify_method"] == "local_llm"
        assert rows[0]["verify_score"] == pytest.approx(0.9)
        assert rows[0]["verify_note"] == "ok"
        assert rows[1]["verified"] == -1
        assert rows[1]["verify_score"] == pytest.approx(0.1)
        assert rows[1]["verify_note"] == "off"

    @pytest.mark.asyncio
    async def test_verify_is_idempotent_does_not_recheck(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        first = await cache.verify_pending(10, QueueJudge([True]))
        assert first["checked"] == 1

        second = await cache.verify_pending(10, QueueJudge([]))
        assert second["checked"] == 0
        assert second["ok"] == 0
        assert _count_rows(db_path, "verified = 1") == 1

    @pytest.mark.asyncio
    async def test_verify_respects_sample_size(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        variants = [
            _PROMPT_A_VARIANT,
            "France capital what the of is?",
            "capital of France what is the?",
            "the of France capital is what?",
            "what the capital is of France?",
        ]
        for variant in variants:
            assert await _hit_lookup(cache, variant) is not None
        assert _count_rows(db_path) == 5

        judge = QueueJudge([True, True])
        result = await cache.verify_pending(2, judge)

        assert result["checked"] == 2
        assert len(judge.calls) == 2
        assert _count_rows(db_path, "verified = 0") == 3

    @pytest.mark.asyncio
    async def test_judge_exception_marks_verify_error(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        class ExplodingJudge:
            def __call__(self, prompt: str, response: str) -> tuple[bool, float, str]:
                raise RuntimeError("judge offline")

        result = await cache.verify_pending(10, ExplodingJudge())

        assert result["checked"] == 1
        assert result["error"] == 1
        assert result["ok"] == 0
        assert result["mismatch"] == 0

        rows = _fetch_log_rows(db_path)
        assert rows[0]["verified"] == 2
        assert rows[0]["verify_note"] is not None
        assert "RuntimeError" in rows[0]["verify_note"]
        assert rows[0]["verify_ts"] is not None

    @pytest.mark.asyncio
    async def test_verify_with_no_pending_returns_zeros(self, db_path: str) -> None:
        cache = SemanticCache(db_path, hit_log_enabled=True)
        judge = QueueJudge([])

        result = await cache.verify_pending(10, judge)

        assert result["checked"] == 0
        assert len(judge.calls) == 0

    @pytest.mark.asyncio
    async def test_async_judge_is_supported(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        class AsyncJudge:
            async def __call__(
                self, prompt: str, response: str
            ) -> tuple[bool, float | None, str | None]:
                return True, 0.77, "async ok"

        result = await cache.verify_pending(10, AsyncJudge())
        assert result["ok"] == 1
        rows = _fetch_log_rows(db_path)
        assert rows[0]["verified"] == 1
        assert rows[0]["verify_score"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# (f) similarity buckets
# ---------------------------------------------------------------------------


class TestVerifyBuckets:
    @pytest.mark.asyncio
    async def test_buckets_group_by_similarity_range(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "hit_log_buckets.db")
        cache = SemanticCache(db_path, threshold=0.80, hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A, embedding=[1.0, 0.0, 0.0])

        for cosine in (0.82, 0.87, 0.92, 0.97):
            hit = await cache.lookup(
                _unit(cosine),
                model="gpt-4o",
                tier=Tier.T2,
                temperature=0.2,
                top_p=0.9,
                max_tokens=100,
            )
            assert hit is not None

        # Rows are selected most-recent-first: 0.97, 0.92, 0.87, 0.82.
        # Plan: the two top buckets verify ok, the two bottom ones mismatch.
        judge = QueueJudge([True, True, False, False])
        result = await cache.verify_pending(10, judge)

        assert result["checked"] == 4
        assert result["ok"] == 2
        assert result["mismatch"] == 2
        assert result["error"] == 0
        assert result["buckets"] == {
            "0.80-0.85": {"checked": 1, "ok": 0, "mismatch": 1},
            "0.85-0.90": {"checked": 1, "ok": 0, "mismatch": 1},
            "0.90-0.95": {"checked": 1, "ok": 1, "mismatch": 0},
            "0.95+": {"checked": 1, "ok": 1, "mismatch": 0},
        }

    @pytest.mark.asyncio
    async def test_judge_error_counts_in_bucket_checked_only(
        self, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "hit_log_bucket_err.db")
        cache = SemanticCache(db_path, threshold=0.80, hit_log_enabled=True)
        await _store_default(cache, _PROMPT_A, embedding=[1.0, 0.0, 0.0])
        hit = await cache.lookup(
            _unit(0.97),
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert hit is not None

        class ExplodingJudge:
            def __call__(self, prompt: str, response: str) -> tuple[bool, float, str]:
                raise ValueError("boom")

        result = await cache.verify_pending(10, ExplodingJudge())

        assert result["buckets"]["0.95+"] == {"checked": 1, "ok": 0, "mismatch": 0}
        assert result["error"] == 1


# ---------------------------------------------------------------------------
# (i) hit_log_enabled=False opts out
# ---------------------------------------------------------------------------


class TestHitLogDisabled:
    @pytest.mark.asyncio
    async def test_disabled_writes_no_row_and_keeps_legacy_stats(
        self, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "hit_log_off.db")
        cache = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), hit_log_enabled=False
        )
        await _store_default(cache, _PROMPT_A)

        hit = await _hit_lookup(cache)

        assert hit is not None
        assert _fetch_log_rows(db_path) == []
        stats = cache.stats()
        assert stats == {
            "semantic_hits": 1,
            "semantic_misses": 0,
            "semantic_unavailable": 0,
        }


# ---------------------------------------------------------------------------
# (j) config knobs
# ---------------------------------------------------------------------------


class TestConfigKnobs:
    def test_defaults(self) -> None:
        config = SemanticCacheConfig()

        assert config.hit_log_enabled is True
        assert config.verify_sample_size == 20
        assert config.verify_judge_base_url == "http://127.0.0.1:11434"
        assert config.verify_judge_model == "glm-5.2"
        assert config.verify_judge_timeout_seconds == pytest.approx(5.0)

    def test_accepts_overrides(self) -> None:
        config = SemanticCacheConfig(
            hit_log_enabled=False,
            verify_sample_size=5,
            verify_judge_base_url="http://ollama.internal:11434",
            verify_judge_model="qwen3",
            verify_judge_timeout_seconds=1.5,
        )

        assert config.hit_log_enabled is False
        assert config.verify_sample_size == 5
        assert config.verify_judge_base_url == "http://ollama.internal:11434"
        assert config.verify_judge_model == "qwen3"
        assert config.verify_judge_timeout_seconds == pytest.approx(1.5)

    def test_rejects_non_positive_sample_size(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SemanticCacheConfig(verify_sample_size=0)
        with pytest.raises(ValidationError):
            SemanticCacheConfig(verify_judge_timeout_seconds=0.0)

    def test_env_nested_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMROUTER_SEMANTIC_CACHE__VERIFY_SAMPLE_SIZE", "7")
        monkeypatch.setenv("LLMROUTER_SEMANTIC_CACHE__HIT_LOG_ENABLED", "false")

        settings = Settings()

        assert settings.semantic_cache.verify_sample_size == 7
        assert settings.semantic_cache.hit_log_enabled is False


# ---------------------------------------------------------------------------
# OllamaJudge: native /api/chat, tolerant yes/no parsing
# ---------------------------------------------------------------------------


class TestOllamaJudge:
    @staticmethod
    def _judge(
        content: str, captured: dict[str, Any] | None = None
    ) -> OllamaJudge:
        def handler(request: httpx.Request) -> httpx.Response:
            if captured is not None:
                captured["url"] = str(request.url)
                captured["payload"] = json.loads(request.read())
            return httpx.Response(200, json={"message": {"content": content}})

        return OllamaJudge(transport=httpx.MockTransport(handler))

    @pytest.mark.asyncio
    async def test_parses_yes_with_reason(self) -> None:
        judge = self._judge("YES — the cached answer directly answers the question.")

        verdict, score, note = await judge("What is 2+2?", "4")

        assert verdict is True
        assert score is None
        assert note is not None and "directly answers" in note

    @pytest.mark.asyncio
    async def test_parses_no_case_insensitive(self) -> None:
        judge = self._judge("No. The answer is about a different city.")

        verdict, score, note = await judge("What is 2+2?", "Paris")

        assert verdict is False
        assert note is not None

    @pytest.mark.asyncio
    async def test_unparseable_reply_raises(self) -> None:
        judge = self._judge("Maybe, it depends on context.")

        with pytest.raises(ValueError):
            await judge("q", "a")

    @pytest.mark.asyncio
    async def test_request_payload_matches_native_contract(self) -> None:
        captured: dict[str, Any] = {}
        judge = self._judge("yes", captured)

        await judge("question text", "response text")

        assert captured["url"].endswith("/api/chat")
        payload = captured["payload"]
        assert payload["model"] == "glm-5.2"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0
        assert payload["messages"][-1]["role"] == "user"
        assert "question text" in payload["messages"][-1]["content"]
        assert "response text" in payload["messages"][-1]["content"]


class TestHitLogRetention:
    """R2: hit-log rows older than ``hit_log_retention_days`` are purged."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "retention.db")

    @pytest.mark.asyncio
    async def test_purge_deletes_old_rows_keeps_recent(self, db_path: str) -> None:
        cache = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True,
            hit_log_retention_days=45,
        )
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        # backdate the single row to 60 days ago
        import time as _time
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE semantic_cache_hit_log SET ts = ?",
                (_time.time() - 60 * 86400,),
            )
            conn.commit()
        assert _count_rows(db_path) == 1

        purged = cache.purge_expired_hit_log()
        assert purged == 1
        assert _count_rows(db_path) == 0

    @pytest.mark.asyncio
    async def test_purge_keeps_rows_within_window(self, db_path: str) -> None:
        cache = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True,
            hit_log_retention_days=45,
        )
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)
        # row ts is "now" — within 45d window
        assert _count_rows(db_path) == 1
        purged = cache.purge_expired_hit_log()
        assert purged == 0
        assert _count_rows(db_path) == 1

    @pytest.mark.asyncio
    async def test_purge_zero_retention_is_noop(self, db_path: str) -> None:
        cache = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True,
            hit_log_retention_days=0,
        )
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)
        purged = cache.purge_expired_hit_log()
        assert purged == 0
        assert _count_rows(db_path) == 1

    @pytest.mark.asyncio
    async def test_verify_pending_triggers_purge(self, db_path: str) -> None:
        cache = SemanticCache(
            db_path, embedder=FakeHashEmbedder(), hit_log_enabled=True,
            hit_log_retention_days=45,
        )
        await _store_default(cache, _PROMPT_A)
        await _hit_lookup(cache)

        import time as _time
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE semantic_cache_hit_log SET ts = ?",
                (_time.time() - 60 * 86400,),
            )
            conn.commit()

        # verify_pending should purge before selecting
        result = await cache.verify_pending(10, QueueJudge([]))
        assert result["checked"] == 0  # row was purged, nothing to verify
        assert _count_rows(db_path) == 0
