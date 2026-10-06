"""The vector store: chunks + their embeddings in a local chromadb database."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from rag_server.chunking import Chunk
from rag_server.embeddings import Embedder

COLLECTION = "golu_docs"


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    source: str
    section: str
    index: int
    library: str | None
    version: str | None
    score: float | None = None  # cosine similarity, 1.0 = identical


class DocStore:
    def __init__(self, embedder: Embedder, path: Path | None = None) -> None:
        """`path=None` keeps everything in memory (used by tests and evals)."""
        settings = Settings(anonymized_telemetry=False)  # no usage pings to chroma
        client = (
            chromadb.PersistentClient(path=str(path), settings=settings)
            if path
            else chromadb.EphemeralClient(settings=settings)
        )
        self.embedder = embedder
        # We compute embeddings ourselves, so chroma's own embedding function is off.
        self._col = client.get_or_create_collection(
            COLLECTION,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": embedder.name},
        )
        built_with = (self._col.metadata or {}).get("embedder")
        if built_with and built_with != embedder.name and self._col.count():
            raise RuntimeError(
                f"The index was built with embedder '{built_with}', not '{embedder.name}'. "
                "Delete the index directory and re-ingest."
            )

    def count(self) -> int:
        return self._col.count()

    def replace_source(self, source: str, chunks: list[Chunk]) -> None:
        """Remove a document's old chunks, then add its new ones."""
        self._col.delete(where={"source": source})
        if not chunks:
            return
        self._col.upsert(
            ids=[_chunk_id(c) for c in chunks],
            embeddings=self.embedder.embed([c.embedding_text() for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )

    def query(self, text: str, k: int, library: str | None = None) -> list[StoredChunk]:
        if self.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=self.embedder.embed([text]),
            n_results=min(k, self.count()),
            where={"library": library} if library else None,
        )
        return [
            _stored(id_, doc, meta, 1.0 - dist)
            for id_, doc, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]

    def by_source(self, source: str) -> list[StoredChunk]:
        res = self._col.get(where={"source": source})
        chunks = [
            _stored(i, d, m, None)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]
        return sorted(chunks, key=lambda c: c.index)

    def sources(self) -> list[str]:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m["source"]) for m in metas})


def _chunk_id(c: Chunk) -> str:
    return hashlib.sha1(f"{c.source}#{c.index}".encode()).hexdigest()[:16]


def _metadata(c: Chunk) -> dict[str, Any]:
    # chroma metadata values can't be None, so "unknown" is stored as "".
    return {
        "source": c.source,
        "section": c.section,
        "index": c.index,
        "title": c.title,
        "library": c.library or "",
        "version": c.version or "",
    }


def _stored(id_: str, doc: str | None, meta: Any, score: float | None) -> StoredChunk:
    return StoredChunk(
        id=id_,
        text=doc or "",
        source=str(meta["source"]),
        section=str(meta["section"]),
        index=int(meta["index"]),
        library=str(meta["library"]) or None,
        version=str(meta["version"]) or None,
        score=score,
    )
