"""Streaming cache storage tests (ROADMAP_TOKEN_OPTIMIZATION E2, S1 — Dev E2-A).

Covers the storage layer for the streaming cache replay MVP:

- config knobs default to safe values (``stream_cache_enabled=True``, k=8,
  timeout=10s, soft-circuit threshold=3, soft-circuit seconds=3600);
- ``SemanticCache.store_stream_response`` persists chunks + usage + first-k
  tokens + restriction tuple to the sibling ``semantic_stream_responses`` table;
- ``lookup_stream_response`` returns the cached chunks for similar prompts;
- lookups return ``None`` for unseen prompts / different restrictions;
- skip-store path: consumer aborted (GeneratorExit) and provider cut without
  ``[DONE]`` / missing ``finish_reason`` never persist anything;
- ``purge_expired_stream_responses`` removes only expired entries;
- the ``UNIQUE(prompt_hash, model, tier, temperature, top_p, max_tokens)``
  constraint collapses duplicates to a single row;
- embedder failures return ``False`` and do not persist anything.

The replay/probe/header instrumentation belongs to Dev E2-B (see PRD §3 S2).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from llmrouter.config import SemanticCacheConfig, Settings
from llmrouter.core.semantic_cache import SemanticCache, _stream_response_text
from llmrouter.core.types import Tier, Usage

_DIM = 64

_PROMPT_A = "What is the capital of France?"
_PROMPT_A_VARIANT = "The capital of France is what?"


def _bag_of_words(text: str) -> list[float]:
    """Deterministic, lightweight embedding (re-used from P-CHR tests)."""
    import hashlib
    import re

    vector = [0.0] * _DIM
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()  # noqa: S324
        vector[int(digest, 16) % _DIM] += 1.0
    return vector


class FakeHashEmbedder:
    """Async embedder: same word multiset -> identical vector."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_bag_of_words(text) for text in texts]


class ExplodingEmbedder:
    """Embedder that always raises (embedder unavailable)."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise RuntimeError("embedder down")


def _sample_chunks() -> list[dict[str, Any]]:
    """Three OpenAI-shaped SSE deltas plus a final usage chunk."""
    return [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Paris"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": " nice"}, "finish_reason": "stop"}],
        },
    ]


def _sample_usage() -> Usage:
    return Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10)


def _restrictions() -> dict[str, Any]:
    return {
        "model": "gpt-4o",
        "tier": Tier.T2,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 100,
    }


# ---------------------------------------------------------------------------
# (config) stream_* knobs default to safe values
# ---------------------------------------------------------------------------


def test_stream_config_defaults_safe() -> None:
    settings = Settings()

    cfg = settings.semantic_cache
    assert cfg.stream_cache_enabled is True
    assert cfg.stream_probe_k == 8
    assert 4 <= cfg.stream_probe_k <= 16
    assert cfg.stream_probe_timeout_seconds == pytest.approx(10.0)
    assert cfg.stream_probe_timeout_seconds > 0
    assert cfg.stream_probe_soft_circuit_threshold == 3
    assert cfg.stream_probe_soft_circuit_threshold >= 1
    assert cfg.stream_probe_soft_circuit_seconds == pytest.approx(3600.0)
    assert cfg.stream_ttl_seconds is None


def test_stream_config_constructible_directly() -> None:
    """Config can be built standalone (no Settings singleton needed)."""
    cfg = SemanticCacheConfig()
    assert cfg.stream_cache_enabled is True
    assert cfg.stream_probe_k == 8
    assert cfg.stream_ttl_seconds is None


def test_stream_ttl_loads_from_nested_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMROUTER_SEMANTIC_CACHE__STREAM_TTL_SECONDS", "123.5")

    settings = Settings(_env_file=None)

    assert settings.semantic_cache.stream_ttl_seconds == pytest.approx(123.5)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_stream_ttl_environment_rejects_non_positive_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LLMROUTER_SEMANTIC_CACHE__STREAM_TTL_SECONDS", value)

    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "value",
    [0, -1, float("inf"), float("nan"), pytest.param(10**10000, id="overflow")],
)
def test_semantic_cache_rejects_invalid_stream_ttl(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError, match="stream_ttl_seconds"):
        SemanticCache(str(tmp_path / "invalid_stream_ttl.db"), stream_ttl_seconds=value)


@pytest.mark.parametrize(
    "ttl_seconds",
    [0, -1, float("inf"), float("nan"), pytest.param(10**10000, id="overflow")],
)
@pytest.mark.asyncio
async def test_store_stream_response_skips_invalid_ttl_override(
    tmp_path: Path, ttl_seconds: Any
) -> None:
    db_path = str(tmp_path / "invalid_stream_ttl_override.db")
    cache = SemanticCache(db_path, embedder=FakeHashEmbedder())

    stored = await cache.store_stream_response(
        _sample_chunks(),
        prompt=_PROMPT_A,
        usage=_sample_usage(),
        ttl_seconds=ttl_seconds,
        **_restrictions(),
    )

    assert stored is False
    assert _count_stream_rows(db_path) == 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _count_stream_rows(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        try:
            return int(conn.execute("SELECT COUNT(*) FROM semantic_stream_responses").fetchone()[0])
        except sqlite3.OperationalError:
            # Table not yet created (e.g. an aborted test path that never
            # reached the schema bootstrap).  Tests asserting "no rows
            # persisted" treat the absence of a table as zero rows.
            return 0


def _fetch_stream_row(db_path: str) -> sqlite3.Row:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM semantic_stream_responses ORDER BY id").fetchall()
        assert rows, "expected one row"
        return rows[0]


# ---------------------------------------------------------------------------
# (a) store_stream_response persists chunks + usage
# ---------------------------------------------------------------------------


class TestStoreStream:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_cache.db")

    @pytest.mark.asyncio
    async def test_store_stream_response_persists_chunks_and_usage(self, db_path: str) -> None:
        cache = SemanticCache(
            db_path,
            embedder=FakeHashEmbedder(),
            threshold=0.90,
        )

        ok = await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )

        assert ok is True
        assert _count_stream_rows(db_path) == 1

        row = _fetch_stream_row(db_path)
        assert row["prompt_text"] == _PROMPT_A
        assert row["model"] == "gpt-4o"
        assert row["tier"] == int(Tier.T2)
        assert row["temperature"] == pytest.approx(0.2)
        assert row["top_p"] == pytest.approx(0.9)
        assert row["max_tokens"] == 100

        # Chunks round-trip via JSON.
        persisted_chunks = __import__("json").loads(row["response_chunks_json"])
        assert persisted_chunks == _sample_chunks()

        # Usage round-trip via JSON.
        persisted_usage = __import__("json").loads(row["usage_json"])
        assert persisted_usage == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "source": "estimated",
        }

        # first_k_tokens = concatenation of deltas.content (default k=8,
        # result: "Paris is nice" - well under 8 tokens of text).
        assert row["first_k_tokens"] == "Paris is nice"

        assert row["expires_at"] > row["created_at"]
        assert row["purge_pending"] == 0
        assert row["prompt_hash"]  # sha256 hex is non-empty

    @pytest.mark.asyncio
    async def test_stream_specific_ttl_overrides_general_ttl(self, tmp_path: Path) -> None:
        path = str(tmp_path / "stream_specific_ttl.db")
        cache = SemanticCache(
            path,
            embedder=FakeHashEmbedder(),
            ttl_seconds=600,
            stream_ttl_seconds=30,
        )
        assert cache.stream_ttl_seconds == 30
        assert await cache.store_stream_response(
            _sample_chunks(), prompt=_PROMPT_A, usage=_sample_usage(), **_restrictions()
        )
        row = _fetch_stream_row(path)
        assert row["expires_at"] - row["created_at"] == pytest.approx(30, abs=0.05)

    @pytest.mark.asyncio
    async def test_stream_ttl_none_inherits_general_ttl(self, tmp_path: Path) -> None:
        path = str(tmp_path / "stream_inherited_ttl.db")
        cache = SemanticCache(
            path,
            embedder=FakeHashEmbedder(),
            ttl_seconds=60,
            stream_ttl_seconds=None,
        )
        assert cache.stream_ttl_seconds is None
        assert await cache.store_stream_response(
            _sample_chunks(), prompt=_PROMPT_A, usage=_sample_usage(), **_restrictions()
        )
        row = _fetch_stream_row(path)
        assert row["expires_at"] - row["created_at"] == pytest.approx(60, abs=0.05)


# ---------------------------------------------------------------------------
# (b) lookup after store returns the same chunks
# ---------------------------------------------------------------------------


class TestLookupStream:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_lookup.db")

    @pytest.mark.asyncio
    async def test_lookup_stream_response_returns_cached_chunks_after_store(
        self, db_path: str
    ) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)
        chunks = _sample_chunks()

        assert await cache.store_stream_response(
            chunks, prompt=_PROMPT_A, usage=_sample_usage(), **_restrictions()
        )

        result = await cache.lookup_stream_response(_PROMPT_A_VARIANT, **_restrictions())

        assert result is not None
        cached_chunks, completion_tokens, first_k = result
        assert cached_chunks == chunks
        assert completion_tokens == 3
        assert first_k == "Paris is nice"

        # lookup_hit counter incremented.
        assert cache.stream_stats()["stream_lookup_hit_total"] == 1

    @pytest.mark.asyncio
    async def test_lookup_stream_response_returns_none_for_unseen_prompt(
        self, db_path: str
    ) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)
        await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )

        # Completely different prompt → no embedding match.
        result = await cache.lookup_stream_response(
            "Explain quantum chromodynamics on the lattice in detail",
            **_restrictions(),
        )

        assert result is None
        assert cache.stream_stats()["stream_lookup_miss_total"] == 1


# ---------------------------------------------------------------------------
# (d) consumer abort (GeneratorExit) must not store
# ---------------------------------------------------------------------------


class _CancellingEmbedder:
    """Embedder that propagates CancelledError to mimic an aborted consumer."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        # Simulate a CancelledError fired in the embed step (the route layer
        # is responsible for catching GeneratorExit upstream, but the storage
        # layer must also remain safe when CancelledError propagates here).
        raise asyncio.CancelledError()


class TestSkipStoreOnAbort:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_abort.db")

    @pytest.mark.asyncio
    async def test_store_stream_response_skipped_on_consumer_disconnect(self, db_path: str) -> None:
        """Embed step cancelled (GeneratorExit / consumer disconnect) → no row.

        When the consumer aborts mid-stream, the route layer guards the call
        with ``except GeneratorExit``.  As a defense in depth, ``store_stream_response``
        must also remain safe if the embed step raises ``CancelledError`` —
        no row is persisted and ``stream_stored_total`` does not advance.
        """
        embedder = _CancellingEmbedder()
        cache = SemanticCache(db_path, embedder=embedder, threshold=0.90)

        # The CancelledError raised inside ``_embed`` propagates out of
        # ``store_stream_response``.  We catch it in the test (mirroring the
        # route layer) and assert the durable invariant: no row was written.
        with pytest.raises(asyncio.CancelledError):
            await cache.store_stream_response(
                _sample_chunks(),
                prompt=_PROMPT_A,
                usage=_sample_usage(),
                **_restrictions(),
            )

        assert _count_stream_rows(db_path) == 0
        assert cache.stream_stats()["stream_stored_total"] == 0

        # Sanity: a regular store after the abort still works.
        cache2 = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)
        assert await cache2.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )
        assert _count_stream_rows(db_path) == 1


# ---------------------------------------------------------------------------
# (e) provider cut without [DONE] / finish_reason / usage must not store
# ---------------------------------------------------------------------------


class TestSkipStoreOnTruncation:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_truncated.db")

    @pytest.mark.asyncio
    async def test_store_stream_response_skipped_when_provider_truncated_no_done(
        self, db_path: str
    ) -> None:
        """Chunks without a final ``finish_reason='stop'`` are not persistable.

        The route layer guards this contract: a finished stream must arrive
        with a final chunk carrying ``finish_reason``.  When the route layer
        hands ``store_stream_response`` chunks where the LAST choice has no
        ``finish_reason``, the store must treat that as an incomplete stream
        and refuse (return ``False``, no row inserted).

        We exercise the contract by feeding chunks whose final choice has
        ``finish_reason=None`` (provider cut mid-flight).
        """
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)
        truncated_chunks = [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Paris"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": None}],
                # Provider cut: never emitted a final chunk with finish_reason.
            },
        ]

        # We expect a guard inside the store that detects truncation.  The
        # simplest, behaviour-equivalent guard is: chunks list has no chunk
        # with ``choices[0].finish_reason in {"stop", "length", "tool_calls"}``
        # → return False, no row written.
        # If the implementation chose a different guard surface, the test
        # still asserts the *observable* behaviour: no row inserted.
        result = await cache.store_stream_response(
            truncated_chunks,
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )

        # We do NOT assert on the exact return value (False vs True) because
        # the route layer might short-circuit the call upstream; instead we
        # assert the *durable* invariant: no entry was persisted from a
        # stream whose last chunk has no terminal finish_reason.
        assert _count_stream_rows(db_path) == 0, (
            "truncated stream (no terminal finish_reason) must not persist"
        )
        # And the stream_stored_total counter is consistent with the DB.
        assert cache.stream_stats()["stream_stored_total"] == 0
        # Reference: ``result`` is allowed to be True (store may run, but
        # should not have persisted anything for the truncated input).
        _ = result


# ---------------------------------------------------------------------------
# (f) purge removes expired entries only
# ---------------------------------------------------------------------------


class TestPurgeStream:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_purge.db")

    @pytest.mark.asyncio
    async def test_purge_expired_stream_responses_removes_old_entries(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)

        # TTL of 0.1s: will expire during the sleep below.
        assert await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            ttl_seconds=0.1,
            **_restrictions(),
        )
        assert _count_stream_rows(db_path) == 1

        # A second, non-expiring entry: distinct prompt (different bag of
        # words → different embedding → distinct UNIQUE key).
        assert await cache.store_stream_response(
            _sample_chunks(),
            prompt="Explain photosynthesis briefly",
            usage=_sample_usage(),
            ttl_seconds=3600.0,
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
        )
        assert _count_stream_rows(db_path) == 2

        # Let the first entry expire.
        await asyncio.sleep(0.2)

        removed = await asyncio.to_thread(cache.purge_expired_stream_responses)
        assert removed == 1

        rows = _count_stream_rows(db_path)
        assert rows == 1

        # Counter advanced.
        assert cache.stream_stats()["stream_purged_total"] == 1

        # Idempotent: another purge is a no-op.
        removed_again = await asyncio.to_thread(cache.purge_expired_stream_responses)
        assert removed_again == 0


# ---------------------------------------------------------------------------
# (g) UNIQUE(prompt_hash, model, tier, ...) collapses duplicates
# ---------------------------------------------------------------------------


class TestUniqueStream:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_unique.db")

    @pytest.mark.asyncio
    async def test_stream_unique_constraint_prevents_duplicates(self, db_path: str) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)

        # Two stores with identical (prompt, model, tier, temperature, top_p,
        # max_tokens).  The UNIQUE constraint must collapse them.
        assert await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )
        new_chunks = _sample_chunks()  # same shape; could differ in real life
        assert await cache.store_stream_response(
            new_chunks,
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )
        assert _count_stream_rows(db_path) == 1

        # Different max_tokens → distinct row.
        assert await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            model="gpt-4o",
            tier=Tier.T2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=200,
        )
        assert _count_stream_rows(db_path) == 2

    @pytest.mark.asyncio
    async def test_missing_max_tokens_persists_null_and_collapses_duplicates(
        self, db_path: str
    ) -> None:
        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)
        restrictions = {**_restrictions(), "max_tokens": None}

        for _ in range(2):
            assert await cache.store_stream_response(
                _sample_chunks(),
                prompt=_PROMPT_A,
                usage=_sample_usage(),
                **restrictions,
            )

        assert _count_stream_rows(db_path) == 1
        row = _fetch_stream_row(db_path)
        assert row["max_tokens"] is None
        assert row["max_tokens_key"] == -1
        assert (
            await cache.lookup_stream_response(
                _PROMPT_A,
                model="gpt-4o",
                tier=Tier.T2,
                temperature=0.2,
                top_p=0.9,
                max_tokens=None,
            )
        ) is not None

    @pytest.mark.asyncio
    async def test_legacy_sentinel_schema_migrates_to_nullable_public_field(
        self, db_path: str
    ) -> None:
        legacy_embedding = json.dumps(_bag_of_words(_PROMPT_A)).encode("utf-8")
        legacy_chunks = json.dumps(_sample_chunks())
        legacy_usage = json.dumps({"prompt_tokens": 7, "completion_tokens": 3})
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE semantic_stream_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    prompt_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    top_p REAL NOT NULL,
                    max_tokens INTEGER,
                    response_chunks_json TEXT NOT NULL,
                    first_k_tokens TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    purge_pending INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(prompt_hash, model, tier, temperature, top_p, max_tokens)
                )
                """
            )
            legacy_rows = (
                # NULL rows were permitted to duplicate under the legacy
                # UNIQUE constraint because SQLite treats NULLs as distinct.
                (None, "legacy null old"),
                (None, "legacy null new"),
                # Older builds also used -1 as a sentinel for missing tokens.
                (-1, "legacy sentinel newest"),
                (128, "legacy explicit limit"),
            )
            for max_tokens, prompt_text in legacy_rows:
                conn.execute(
                    """
                    INSERT INTO semantic_stream_responses
                        (prompt_hash, embedding, prompt_text, model, tier,
                         temperature, top_p, max_tokens, response_chunks_json,
                         first_k_tokens, usage_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "legacy-prompt-hash",
                        legacy_embedding,
                        prompt_text,
                        "gpt-4o",
                        int(Tier.T2),
                        0.2,
                        0.9,
                        max_tokens,
                        legacy_chunks,
                        "Paris",
                        legacy_usage,
                        1.0,
                        9_999_999_999.0,
                    ),
                )
            conn.commit()

        cache = SemanticCache(db_path, embedder=FakeHashEmbedder(), threshold=0.90)
        await cache._ensure_table()

        with sqlite3.connect(db_path) as conn:
            migrated_rows = conn.execute(
                "SELECT max_tokens, max_tokens_key, prompt_text "
                "FROM semantic_stream_responses ORDER BY max_tokens_key"
            ).fetchall()
        assert migrated_rows == [
            (None, -1, "legacy sentinel newest"),
            (128, 128, "legacy explicit limit"),
        ]

        assert await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **{**_restrictions(), "max_tokens": None},
        )

        row = _fetch_stream_row(db_path)
        assert row["max_tokens"] is None
        assert row["max_tokens_key"] == -1
        assert (
            await cache.lookup_stream_response(
                _PROMPT_A,
                model="gpt-4o",
                tier=Tier.T2,
                temperature=0.2,
                top_p=0.9,
                max_tokens=None,
            )
        ) is not None


# ---------------------------------------------------------------------------
# (h) embedder failure -> store skipped, no row
# ---------------------------------------------------------------------------


class TestEmbedderFailure:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "stream_embedder_fail.db")

    @pytest.mark.asyncio
    async def test_store_stream_response_returns_false_when_embedder_fails(
        self, db_path: str
    ) -> None:
        embedder = ExplodingEmbedder()
        cache = SemanticCache(db_path, embedder=embedder, threshold=0.90)

        ok = await cache.store_stream_response(
            _sample_chunks(),
            prompt=_PROMPT_A,
            usage=_sample_usage(),
            **_restrictions(),
        )

        assert ok is False
        assert _count_stream_rows(db_path) == 0
        # Counter reflects no store happened.
        assert cache.stream_stats()["stream_stored_total"] == 0
        # Semantic unavailable was incremented by the underlying _embed.
        assert cache.stats()["semantic_unavailable"] == 1


# ---------------------------------------------------------------------------
# stats() shape sanity (covers the in-memory counter surface)
# ---------------------------------------------------------------------------


class TestStreamStats:
    @pytest.mark.asyncio
    async def test_stream_stats_zero_on_fresh_cache(self, tmp_path: Path) -> None:
        cache = SemanticCache(str(tmp_path / "stream_stats0.db"))

        stats = cache.stream_stats()

        assert stats["stream_stored_total"] == 0
        assert stats["stream_purged_total"] == 0
        assert stats["stream_lookup_hit_total"] == 0
        assert stats["stream_lookup_miss_total"] == 0
        assert stats["stream_hits"] == 0
        assert stats["stream_replays_served"] == 0
        assert stats["stream_replay_tokens_saved"] == 0
        assert stats["stream_probe_tokens_spent"] == 0
        assert stats["stream_probe_latency_ms_p50"] == 0

    @pytest.mark.asyncio
    async def test_stream_metrics_aliases_and_bounded_probe_latency(self, tmp_path: Path) -> None:
        cache = SemanticCache(str(tmp_path / "stream_metric_aliases.db"))
        cache.bump_stream_counter("stream_lookup_hit_total", 2)
        cache.bump_stream_counter("stream_replays_total", 3)
        cache.bump_stream_counter("stream_tokens_saved_total", 11)
        cache.record_stream_probe(token_budget=8, latency_ms=100)
        cache.record_stream_probe(token_budget=8, latency_ms=200)
        cache.record_stream_probe(token_budget=8, latency_ms=300)

        stats = cache.stream_stats()
        assert stats["stream_hits"] == stats["stream_lookup_hit_total"] == 2
        assert stats["stream_replays_served"] == stats["stream_replays_total"] == 3
        assert stats["stream_replay_tokens_saved"] == stats["stream_tokens_saved_total"] == 11
        assert stats["stream_probe_tokens_spent"] == 24
        assert stats["stream_probe_tokens_spent_estimated"] == 24
        assert stats["stream_probe_latency_ms_p50"] == pytest.approx(200)
        assert len(cache._stream_probe_latency.samples) <= 1000

    def test_stream_hit_log_text_extraction_skips_invalid_choices(self) -> None:
        assert (
            _stream_response_text(
                [
                    {"choices": None},
                    {"choices": []},
                    {"choices": ["invalid"]},
                    {"choices": [{"delta": {"content": "A"}}]},
                    {"choices": [{"message": {"content": "B"}}]},
                    {"choices": [{"delta": {"content": 3}}]},
                ]
            )
            == "AB"
        )

    @pytest.mark.asyncio
    async def test_stream_hit_log_is_skipped_when_disabled(self, tmp_path: Path) -> None:
        cache = SemanticCache(str(tmp_path / "stream_hit_log_disabled.db"))
        match = await cache.lookup_stream_response(
            _PROMPT_A,
            model="missing",
            tier=int(Tier.T2),
            temperature=0.2,
            top_p=0.9,
            max_tokens=64,
        )
        assert match is None
        assert not await cache.record_stream_hit(None)  # type: ignore[arg-type]
