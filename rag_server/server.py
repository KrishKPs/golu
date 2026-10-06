"""MCP server exposing the documentation index.

Tools:
- search_docs(query, k, library): the most relevant doc chunks
- get_doc(source, section): a full document or section

Both return structured results (source, section, version, text) so the client
can show citations.
"""

from functools import lru_cache

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from rag_server.retrieve import Retriever
from rag_server.store import StoredChunk


class DocChunk(BaseModel):
    source: str
    section: str
    version: str | None = None
    library: str | None = None
    score: float | None = None
    text: str


class DocResults(BaseModel):
    results: list[DocChunk]
    note: str


@lru_cache(maxsize=1)
def default_retriever() -> Retriever:
    # Loaded on first use so the server starts instantly.
    from rag_server.config import db_path, models_dir
    from rag_server.embeddings import LocalEmbedder
    from rag_server.store import DocStore

    return Retriever(DocStore(LocalEmbedder(models_dir()), db_path()))


def _to_model(c: StoredChunk) -> DocChunk:
    score = round(c.score, 3) if c.score is not None else None
    return DocChunk(
        source=c.source,
        section=c.section,
        version=c.version,
        library=c.library,
        score=score,
        text=c.text,
    )


def build_server(retriever: Retriever | None = None) -> MCPServer:
    """`retriever` is injectable for tests; the real one loads lazily."""

    def get() -> Retriever:
        return retriever or default_retriever()

    server = MCPServer(
        "golu-docs",
        instructions="Search the documentation indexed for this project. Cite source and section.",
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True), structured_output=True)
    def search_docs(query: str, k: int = 5, library: str | None = None) -> DocResults:
        """Search the indexed library/API documentation.

        Returns the most relevant chunks, each with its source file, section heading,
        and library version. Use specific queries naming the function or concept, e.g.
        "Store.put ttl_seconds parameter". An empty result means the docs don't cover it.
        `library` restricts results to one library's docs.
        """
        hits = get().search(query, k, library)
        if not hits:
            note = "No relevant documentation found. The indexed docs do not cover this."
        else:
            note = f"{len(hits)} chunk(s). Cite as [source § section]."
        return DocResults(results=[_to_model(h) for h in hits], note=note)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True), structured_output=True)
    def get_doc(source: str, section: str | None = None) -> DocResults:
        """Read a whole indexed document, or one section of it, by its source name.

        Use after search_docs when you need the surrounding context of a chunk.
        `source` and `section` are exactly as search_docs returned them.
        """
        chunks = get().get_doc(source, section)
        if not chunks:
            known = ", ".join(get().store.sources()[:30]) or "none"
            note = f"No document/section found for that source. Indexed sources: {known}"
        else:
            note = f"{len(chunks)} chunk(s) in document order."
        return DocResults(results=[_to_model(c) for c in chunks], note=note)

    return server
