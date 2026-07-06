"""Shared test fixtures for the LLMrouter test suite."""

from __future__ import annotations

import logging
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _restore_logging_after_tests() -> Any:
    """Restore logging configuration before and after each test.

    Some tests call ``setup_logging`` or trigger ``_ensure_runtime_logging``
    (via ``build_app()`` at module import time) which modify the root logger
    handlers and disable propagation on the ``llmrouter`` logger, breaking
    pytest's ``caplog`` fixture for subsequent tests.
    """
    root = logging.getLogger()
    llmrouter_logger = logging.getLogger("llmrouter")

    # Save original state
    original_root_level = root.level
    original_root_handlers = list(root.handlers)
    original_llmrouter_level = llmrouter_logger.level
    original_llmrouter_propagate = llmrouter_logger.propagate
    original_llmrouter_handlers = list(llmrouter_logger.handlers)

    # SETUP: ensure propagation is enabled and root has a handler so caplog works
    llmrouter_logger.propagate = True

    yield

    # TEARDOWN: restore everything
    root.setLevel(original_root_level)
    root.handlers = original_root_handlers
    llmrouter_logger.setLevel(original_llmrouter_level)
    llmrouter_logger.propagate = original_llmrouter_propagate
    llmrouter_logger.handlers = original_llmrouter_handlers