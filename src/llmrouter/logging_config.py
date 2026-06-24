"""Logging configuration for LLMrouter.

Provides structured logging with debug support that can be toggled via
a ``--debug`` CLI flag or the ``LLMROUTER_DEBUG`` environment variable.

Usage::

    from llmrouter.logging_config import setup_logging, get_logger

    setup_logging(debug=True)
    logger = get_logger(__name__)
    logger.debug("This only shows in debug mode")
"""

from __future__ import annotations

import logging
import os
import sys


# ANSI color codes for terminal output
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(debug: bool | None = None) -> None:
    """Configure logging for the entire application.

    Args:
        debug: If True, set log level to DEBUG. If None, checks the
               ``LLMROUTER_DEBUG`` environment variable.
    """
    if debug is None:
        debug = os.environ.get("LLMROUTER_DEBUG", "").lower() in ("1", "true", "yes")

    level = logging.DEBUG if debug else logging.INFO

    # Format: [LEVEL] message
    fmt = f"[%(levelname)s] %(name)s: %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColoredFormatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-init
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)

    # Silence noisy libraries unless in debug mode
    noisy_loggers = ["httpcore", "httpx", "urllib3", "asyncio"]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.DEBUG if debug else logging.WARNING)

    # Uvicorn access logs — keep at WARNING in normal mode, DEBUG in debug mode
    logging.getLogger("uvicorn.access").setLevel(logging.DEBUG if debug else logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)