"""Application entrypoint."""

from __future__ import annotations

import argparse

import uvicorn

from llmrouter.config import get_settings
from llmrouter.logging_config import setup_logging
from llmrouter.runtime import build_app

app = build_app()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLMrouter — Multi-model LLM gateway with intelligent routing",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        default=False,
        help="Enable debug mode with detailed request/routing/decision logging.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: from config, usually 0.0.0.0).",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Server port (default: from config, usually 12345).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on file changes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the development server."""
    args = _parse_args()
    settings = get_settings()

    # Configure logging based on --debug flag
    setup_logging(debug=args.debug)
    if args.debug:
        import logging
        logging.getLogger("llmrouter").info("Debug mode ENABLED — detailed logging active")

    host = args.host or settings.server.host
    port = args.port or settings.server.port
    reload = args.reload or settings.debug

    uvicorn.run(
        "llmrouter.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()