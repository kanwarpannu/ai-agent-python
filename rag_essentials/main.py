from __future__ import annotations

import argparse

from .config import get_settings
from .logging_config import configure_logging
from .rag_service import RAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable RAG module using existing lib/ abstractions")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("health", help="Check Chroma connectivity and resolved PDF path")
    subcommands.add_parser("ingest", help="Load and index the configured PDF")

    ask = subcommands.add_parser("ask", help="Ask a question against the indexed collection")
    ask.add_argument("question", help="Question to ask")
    ask.add_argument("--json", action="store_true", help="Print full JSON response")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    service = RAGService(settings)

    if args.command == "health":
        print(f"Chroma heartbeat: {service.vector_store.heartbeat()}")
        print(f"Resolved PDF path: {settings.app.pdf_path}")
        print(f"Collection: {settings.app.collection_name}")
        return

    if args.command == "ingest":
        count = service.ingest()
        print(f"Indexed {count} chunks into collection '{settings.app.collection_name}'")
        return

    if args.command == "ask":
        result = service.ask(args.question)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print("\nAnswer:\n")
            print(result.answer)
            print("\nSources:\n")
            for idx, hit in enumerate(result.context_hits, start=1):
                source = hit.metadata.get("source", "unknown")
                page = hit.metadata.get("page", "?")
                print(f"{idx}. source={source}, page={page}, distance={hit.distance}")
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
