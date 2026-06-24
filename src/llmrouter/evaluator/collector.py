"""Observation collection and SQLite persistence."""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
from collections import deque
from pathlib import Path

from llmrouter.core.types import RoutingGrade
from llmrouter.evaluator.types import RoutingObservation, RoutingReview, TrainingExample


class ObservationCollector:
    """Non-blocking in-memory buffer with SQLite persistence."""

    def __init__(
        self,
        db_path: str = "data/llmrouter.db",
        buffer_size: int = 100,
        sample_rate: float = 1.0,
    ) -> None:
        self._db_path = db_path
        self._buffer: deque[RoutingObservation] = deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()
        self._sample_rate = min(max(sample_rate, 0.0), 1.0)

    def record(self, observation: RoutingObservation) -> None:
        """Add an observation to the in-memory buffer without awaiting."""
        if self._sample_rate < 1.0 and random.random() > self._sample_rate:
            return
        self._buffer.append(observation)

    async def flush(self) -> list[int]:
        """Persist buffered observations and return inserted row IDs."""
        async with self._lock:
            observations = list(self._buffer)
            self._buffer.clear()
        if not observations:
            return []

        await self.initialize()
        ids: list[int] = []
        with sqlite3.connect(self._db_path) as db:
            for item in observations:
                cursor = db.execute(
                    """
                    INSERT INTO observations (
                        prompt, chosen_model, response, latency_ms, cost_usd,
                        prompt_tokens, completion_tokens, scorer_score,
                        scorer_tier, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.prompt,
                        item.chosen_model,
                        item.response,
                        item.latency_ms,
                        item.cost_usd,
                        item.prompt_tokens,
                        item.completion_tokens,
                        item.scorer_score,
                        item.scorer_tier,
                        json.dumps(item.metadata, sort_keys=True),
                    ),
                )
                ids.append(int(cursor.lastrowid))
            db.commit()
        return ids

    async def save_review(self, review: RoutingReview) -> None:
        """Persist a judge/grader review for one observation."""
        await self.initialize()
        with sqlite3.connect(self._db_path) as db:
            db.execute(
                """
                INSERT INTO reviews (
                    observation_id, relevance, accuracy, completeness, concision,
                    safety, quality_overall, grade, suggested_model, rationale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.observation_id,
                    review.quality.relevance,
                    review.quality.accuracy,
                    review.quality.completeness,
                    review.quality.concision,
                    review.quality.safety,
                    review.quality.overall,
                    review.grade.value,
                    review.suggested_model,
                    review.rationale or review.quality.rationale,
                ),
            )
            db.commit()

    async def get_pending_observations(
        self, limit: int = 50
    ) -> list[tuple[int, RoutingObservation]]:
        """Return observations that do not have reviews yet."""
        await self.initialize()
        with sqlite3.connect(self._db_path) as db:
            rows = db.execute(
                """
                SELECT o.id, o.prompt, o.chosen_model, o.response, o.latency_ms,
                       o.cost_usd, o.prompt_tokens, o.completion_tokens,
                       o.scorer_score, o.scorer_tier, o.metadata_json
                FROM observations o
                LEFT JOIN reviews r ON r.observation_id = o.id
                WHERE r.id IS NULL
                ORDER BY o.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(int(row[0]), _row_to_observation(row[1:])) for row in rows]

    async def get_training_data(self, limit: int = 1000) -> list[TrainingExample]:
        """Load reviewed examples suitable for scorer calibration."""
        await self.initialize()
        with sqlite3.connect(self._db_path) as db:
            rows = db.execute(
                """
                SELECT o.prompt, o.chosen_model, r.grade, r.quality_overall,
                       o.scorer_score, o.scorer_tier
                FROM observations o
                JOIN reviews r ON r.observation_id = o.id
                ORDER BY r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            TrainingExample(
                prompt=str(row[0]),
                chosen_model=str(row[1]),
                grade=RoutingGrade(str(row[2])),
                quality_overall=float(row[3]),
                scorer_score=row[4],
                scorer_tier=row[5],
            )
            for row in rows
        ]

    async def initialize(self) -> None:
        """Create database tables if needed."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    chosen_model TEXT NOT NULL,
                    response TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    scorer_score REAL,
                    scorer_tier INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id INTEGER NOT NULL REFERENCES observations(id),
                    relevance INTEGER NOT NULL,
                    accuracy INTEGER NOT NULL,
                    completeness INTEGER NOT NULL,
                    concision INTEGER NOT NULL,
                    safety INTEGER NOT NULL,
                    quality_overall REAL NOT NULL,
                    grade TEXT NOT NULL,
                    suggested_model TEXT,
                    rationale TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            db.commit()


def _row_to_observation(row: tuple[object, ...]) -> RoutingObservation:
    metadata = json.loads(str(row[9] or "{}"))
    if not isinstance(metadata, dict):
        metadata = {}
    return RoutingObservation(
        prompt=str(row[0]),
        chosen_model=str(row[1]),
        response=str(row[2]),
        latency_ms=float(row[3]),
        cost_usd=float(row[4]),
        prompt_tokens=int(row[5]),
        completion_tokens=int(row[6]),
        scorer_score=float(row[7]) if row[7] is not None else None,
        scorer_tier=int(row[8]) if row[8] is not None else None,
        metadata={str(key): str(value) for key, value in metadata.items()},
    )
