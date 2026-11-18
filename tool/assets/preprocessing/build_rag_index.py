"""Command-line entry point for preparing the engineering RAG index."""

import sys

from tool.rag.engineering_rag import main as engineering_rag_main


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    return engineering_rag_main(["index", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
