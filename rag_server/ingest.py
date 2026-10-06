"""Index documents: load -> chunk -> embed -> store.

    uv run python -m rag_server.ingest <file-or-folder> [--library NAME] [--version X.Y]

Re-ingesting a file replaces its old chunks.
"""

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rag_server.chunking import Chunk, chunk_markdown, chunk_plain_text
from rag_server.store import DocStore

MARKDOWN = {".md", ".markdown", ".mdx"}
PLAIN = {".txt", ".rst"}


@dataclass
class IngestStats:
    files: int = 0
    chunks: int = 0
    skipped: int = 0


def find_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in MARKDOWN | PLAIN and not p.name.startswith("."):
            yield p


def chunk_file(
    file: Path, source: str, library: str | None = None, version: str | None = None
) -> list[Chunk]:
    text = file.read_text(encoding="utf-8", errors="replace")
    if file.suffix.lower() in MARKDOWN:
        return chunk_markdown(text, source, library=library, version=version)
    return chunk_plain_text(text, source, library=library, version=version)


def ingest(
    path: Path, store: DocStore, library: str | None = None, version: str | None = None
) -> IngestStats:
    stats = IngestStats()
    base = path if path.is_dir() else path.parent
    for file in find_files(path):
        if file.suffix.lower() not in MARKDOWN | PLAIN:
            stats.skipped += 1
            continue
        # Sources are stored relative to the folder you ingested, e.g. "api-reference.md".
        source = file.relative_to(base).as_posix()
        chunks = chunk_file(file, source, library, version)
        store.replace_source(chunks[0].source if chunks else source, chunks)
        stats.files += 1
        stats.chunks += len(chunks)
    return stats


def main(argv: list[str] | None = None) -> int:
    from rag_server.config import db_path, models_dir
    from rag_server.embeddings import LocalEmbedder

    parser = argparse.ArgumentParser(description="Index documentation for Golu.")
    parser.add_argument("path", type=Path, help="A file or folder of .md/.txt/.rst docs")
    parser.add_argument("--library", help="Library name to tag chunks with")
    parser.add_argument("--version", help="Library version, if not in the files' front matter")
    args = parser.parse_args(argv)
    if not args.path.exists():
        print(f"Not found: {args.path}", file=sys.stderr)
        return 1

    store = DocStore(LocalEmbedder(models_dir()), db_path())
    stats = ingest(args.path, store, args.library, args.version)
    print(
        f"Indexed {stats.files} file(s) into {stats.chunks} chunk(s)"
        + (f", skipped {stats.skipped}" if stats.skipped else "")
        + f". Index now has {store.count()} chunks at {db_path()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
