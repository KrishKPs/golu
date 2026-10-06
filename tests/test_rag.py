import asyncio
from pathlib import Path

import pytest
from mcp import Client

from rag_server.embeddings import HashEmbedder
from rag_server.ingest import ingest
from rag_server.retrieve import Retriever
from rag_server.server import build_server
from rag_server.store import DocStore

CORPUS = Path(__file__).parent.parent / "evals" / "corpus" / "quillstore"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    store = DocStore(HashEmbedder())
    ingest(CORPUS, store)
    return Retriever(store, min_score=0.2)


def test_ingest_stores_metadata(retriever: Retriever) -> None:
    assert "api-reference.md" in retriever.store.sources()
    chunk = retriever.store.by_source("api-reference.md")[0]
    assert chunk.library == "quillstore" and chunk.version == "3.2"


def test_reingest_replaces_chunks(tmp_path: Path) -> None:
    store = DocStore(HashEmbedder())
    doc = tmp_path / "a.md"
    doc.write_text("# A\n\none\n\n## B\n\ntwo\n")
    ingest(tmp_path, store)
    doc.write_text("# A\n\nonly one now\n")
    ingest(tmp_path, store)
    assert [c.text for c in store.by_source("a.md")] == ["# A\n\nonly one now"]


def test_search_finds_relevant_section(retriever: Retriever) -> None:
    hits = retriever.search("ttl_seconds expire put", k=3)
    assert hits and hits[0].section == "API reference > Store.put"
    assert hits[0].score is not None and 0 < hits[0].score <= 1


def test_unrelated_query_returns_nothing(retriever: Retriever) -> None:
    assert retriever.search("banana bread oven recipe", k=5) == []


def test_get_doc_section(retriever: Retriever) -> None:
    chunks = retriever.get_doc("errors.md", "ConflictError")
    assert len(chunks) == 1 and "retry_transaction" in chunks[0].text
    assert retriever.get_doc("nope.md") == []


def test_persistent_store_and_embedder_mismatch(tmp_path: Path) -> None:
    store = DocStore(HashEmbedder(), tmp_path / "db")
    ingest(CORPUS / "cli.md", store)

    class Other(HashEmbedder):
        name = "other"

    with pytest.raises(RuntimeError, match="re-ingest"):
        DocStore(Other(), tmp_path / "db")


def test_mcp_server_search_and_get(retriever: Retriever) -> None:
    async def main() -> None:
        async with Client(build_server(retriever)) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {"search_docs", "get_doc"}
            assert tools["search_docs"].annotations.read_only_hint

            res = await client.call_tool("search_docs", {"query": "scan prefix reverse"})
            assert not res.is_error
            first = res.structured_content["results"][0]
            assert first["source"] == "api-reference.md"
            assert first["section"] == "API reference > Store.scan"
            assert first["version"] == "3.2"

            empty = await client.call_tool("search_docs", {"query": "banana bread oven"})
            assert empty.structured_content["results"] == []
            assert "do not cover" in empty.structured_content["note"]

            doc = await client.call_tool("get_doc", {"source": "cli.md"})
            assert len(doc.structured_content["results"]) >= 6

    asyncio.run(main())
