"""Application entrypoint."""

from __future__ import annotations

import uvicorn

from llmrouter.config import get_settings
from llmrouter.runtime import build_app

app = build_app()


def main() -> None:
    """Run the development server."""
    settings = get_settings()
    uvicorn.run(
        "llmrouter.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
