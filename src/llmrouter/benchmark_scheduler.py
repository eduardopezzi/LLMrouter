"""In-process periodic refresh for the local benchmark score catalog."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from llmrouter.benchmark_catalog import (
    BenchmarkRefreshError,
    RefreshReport,
    refresh_benchmark_catalog,
)
from llmrouter.benchmark_research import BenchmarkResearcher, BenchmarkResearchReport

_LOGGER = logging.getLogger("llmrouter.benchmarks")


class BenchmarkRefreshScheduler:
    """Refresh declared sources in the background without delaying requests."""

    def __init__(
        self,
        *,
        sources_path: str,
        catalog_path: str,
        interval_seconds: float = 15 * 24 * 60 * 60,
        timeout: float = 30.0,
        on_catalog_changed: Callable[[RefreshReport], None] | None = None,
        researcher: BenchmarkResearcher | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.sources_path = sources_path
        self.catalog_path = catalog_path
        self.interval_seconds = interval_seconds
        self.timeout = timeout
        self.on_catalog_changed = on_catalog_changed
        self.researcher = researcher
        self.last_report: RefreshReport | None = None
        self.last_research_report: BenchmarkResearchReport | None = None
        self.last_error: str | None = None

    async def run(self) -> None:
        """Run immediately once, then repeat at the configured interval."""
        while True:
            await self.refresh_once()
            await asyncio.sleep(self.interval_seconds)

    async def refresh_once(self) -> RefreshReport | None:
        """Refresh once in a worker thread, or skip if another worker owns the lock."""
        try:
            report = await asyncio.to_thread(self._refresh_with_lock)
        except BenchmarkRefreshError as exc:
            self.last_error = str(exc)
            _LOGGER.warning("Benchmark refresh failed: %s", exc)
            return None
        except Exception:
            self.last_error = "Unexpected benchmark refresh failure"
            _LOGGER.exception("Unexpected benchmark refresh failure")
            return None
        if report is None:
            _LOGGER.debug("Benchmark refresh is already running in another worker")
            return None

        self.last_report = report
        self.last_error = None
        if report.changed:
            _LOGGER.info(
                "Benchmark catalog updated: %d model(s), %d score(s)",
                report.models_updated,
                report.scores_updated,
            )
            if self.on_catalog_changed is not None:
                self.on_catalog_changed(report)
        else:
            _LOGGER.info("Benchmark catalog is already up to date")
        if self.researcher is not None:
            await self._run_research()
        return report

    async def _run_research(self) -> None:
        try:
            self.last_research_report = await self.researcher.research(
                self.sources_path,
                self.catalog_path,
            )
            _LOGGER.info(
                "Benchmark research proposals written: %d source(s), %d model(s)",
                self.last_research_report.source_proposals,
                self.last_research_report.model_proposals,
            )
        except Exception:
            _LOGGER.exception("LLM-assisted benchmark research failed; catalog was not changed")

    def _refresh_with_lock(self) -> RefreshReport | None:
        lock_path = Path(f"{self.catalog_path}.lock")
        with _exclusive_lock(lock_path) as acquired:
            if not acquired:
                return None
            return refresh_benchmark_catalog(
                self.sources_path,
                self.catalog_path,
                timeout=self.timeout,
            )


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[bool]:
    """Use an advisory non-blocking file lock across Uvicorn workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows has no fcntl.
            yield True
            return

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
