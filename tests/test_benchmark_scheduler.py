from __future__ import annotations

import asyncio

from llmrouter.benchmark_catalog import RefreshReport
from llmrouter.benchmark_scheduler import BenchmarkRefreshScheduler


def test_scheduler_refreshes_in_worker_thread_and_notifies_on_change(monkeypatch, tmp_path) -> None:
    expected = RefreshReport(
        models_updated=1,
        scores_updated=2,
        changed=True,
        output_path=tmp_path / "catalog.yaml",
    )
    notified: list[RefreshReport] = []
    monkeypatch.setattr(
        "llmrouter.benchmark_scheduler.refresh_benchmark_catalog",
        lambda *_args, **_kwargs: expected,
    )
    scheduler = BenchmarkRefreshScheduler(
        sources_path=str(tmp_path / "sources.yaml"),
        catalog_path=str(expected.output_path),
        interval_seconds=1,
        on_catalog_changed=notified.append,
    )

    report = asyncio.run(scheduler.refresh_once())

    assert report == expected
    assert scheduler.last_error is None
    assert notified == [expected]


def test_scheduler_records_refresh_error(monkeypatch, tmp_path) -> None:
    def fail(*_args, **_kwargs):
        from llmrouter.benchmark_catalog import BenchmarkRefreshError

        raise BenchmarkRefreshError("bad source")

    monkeypatch.setattr("llmrouter.benchmark_scheduler.refresh_benchmark_catalog", fail)
    scheduler = BenchmarkRefreshScheduler(
        sources_path=str(tmp_path / "sources.yaml"),
        catalog_path=str(tmp_path / "catalog.yaml"),
        interval_seconds=1,
    )

    assert asyncio.run(scheduler.refresh_once()) is None
    assert scheduler.last_error == "bad source"
