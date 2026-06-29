"""Application entrypoint."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from llmrouter.config import get_settings
from llmrouter.contract_publisher import ContractPublisher
from llmrouter.cross_repository import (
    BreakingChangeDetector,
    ContractRegistry,
    format_contract_changes,
    resolve_project_contract_path,
)
from llmrouter.logging_config import setup_logging
from llmrouter.runtime import build_app, build_registry

app = build_app()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLMrouter — Multi-model LLM gateway with intelligent routing",
    )
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser(
        "export-contracts",
        help="Export a cross-repository contract snapshot.",
    )
    export_parser.add_argument(
        "--models-file",
        type=str,
        default=None,
        help="Model catalog path (default: from config).",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="contracts/llmrouter.contract.json",
        help="Output JSON snapshot path.",
    )
    export_parser.add_argument(
        "--contracts-root",
        type=str,
        default=None,
        help="Shared contracts repository root. When set, output is PROJECT/FILENAME.",
    )
    export_parser.add_argument(
        "--project",
        type=str,
        default="llmrouter",
        help="Project folder inside --contracts-root; matched case-insensitively.",
    )
    export_parser.add_argument(
        "--filename",
        type=str,
        default="llmrouter.contract.json",
        help="Current contract filename inside the project folder.",
    )
    export_parser.add_argument(
        "--service",
        type=str,
        default="llmrouter",
        help="Logical service name in the snapshot.",
    )

    check_parser = subparsers.add_parser(
        "check-contracts",
        help="Fail when a new contract snapshot breaks a previous snapshot.",
    )
    check_parser.add_argument("previous", type=str, help="Previous JSON snapshot path.")
    check_parser.add_argument("current", type=str, help="Current JSON snapshot path.")

    diff_parser = subparsers.add_parser(
        "diff-contracts",
        help="Print contract differences without failing on breaking changes.",
    )
    diff_parser.add_argument("previous", type=str, help="Previous JSON snapshot path.")
    diff_parser.add_argument("current", type=str, help="Current JSON snapshot path.")

    publish_parser = subparsers.add_parser(
        "publish-contracts",
        help="Publish the current contract to the shared GitHub versions repository.",
    )
    publish_parser.add_argument(
        "--models-file",
        type=str,
        default=None,
        help="Model catalog path (default: from config).",
    )
    publish_parser.add_argument(
        "--repo",
        type=str,
        default="https://github.com/Vieli-Tech/phoenix_versions.git",
        help="Shared GitHub versions repository URL.",
    )
    publish_parser.add_argument("--branch", type=str, default="main", help="Repository branch.")
    publish_parser.add_argument(
        "--project",
        type=str,
        default="llmrouter",
        help="Project folder; matched case-insensitively.",
    )
    publish_parser.add_argument(
        "--filename",
        type=str,
        default="llmrouter.contract.json",
        help="Current contract filename inside the project folder.",
    )
    publish_parser.add_argument(
        "--service",
        type=str,
        default="llmrouter",
        help="Logical service name in the snapshot.",
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

    if args.command == "export-contracts":
        registry = build_registry(args.models_file or settings.models_file)
        output = (
            resolve_project_contract_path(
                args.contracts_root,
                args.project,
                args.filename,
                create=True,
            )
            if args.contracts_root
            else args.output
        )
        ContractRegistry(registry=registry, service_name=args.service).write_snapshot(output)
        print(f"Exported contract snapshot to {output}")
        return

    if args.command in {"check-contracts", "diff-contracts"}:
        result = BreakingChangeDetector().compare_files(args.previous, args.current)
        print(format_contract_changes(result))
        if args.command == "check-contracts" and not result.is_compatible:
            sys.exit(1)
        return

    if args.command == "publish-contracts":
        registry = build_registry(args.models_file or settings.models_file)
        result = ContractPublisher(
            repository_url=args.repo,
            branch=args.branch,
            project=args.project,
            filename=args.filename,
            service_name=args.service,
        ).publish(registry)
        if result.changed:
            print(f"Published contract {result.contract_path} at {result.commit_sha}")
        else:
            print(f"Contract already up to date: {result.contract_path}")
        return

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
