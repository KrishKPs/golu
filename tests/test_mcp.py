from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.server import MCPServer

from golu.agent.loop import Agent
from golu.llm.base import ToolCall
from golu.llm.fake import FakeProvider
from golu.mcp.client import MCPManager
from golu.mcp.config import ConfigError, load_servers
from golu.permissions import Permissions
from rag_server.embeddings import HashEmbedder
from rag_server.ingest import ingest
from rag_server.retrieve import Retriever
from rag_server.server import build_server
from rag_server.store import DocStore

CORPUS = Path(__file__).parent.parent / "evals" / "corpus" / "quillstore"


@pytest.fixture(scope="module")
def docs_server() -> MCPServer:
    store = DocStore(HashEmbedder())
    ingest(CORPUS, store)
    return build_server(Retriever(store, min_score=0.2))


@pytest.fixture
def manager(docs_server: MCPServer) -> Iterator[MCPManager]:
    m = MCPManager()
    m.connect("docs", docs_server, "read_only")
    yield m
    m.close()


def test_tools_are_prefixed_and_described(manager: MCPManager) -> None:
    names = sorted(t.name for t in manager.tools)
    assert names == ["docs__get_doc", "docs__search_docs"]
    search = next(t for t in manager.tools if t.name == "docs__search_docs")
    assert "MCP server: docs" in search.description
    assert search.input_schema["required"] == ["query"]


def test_read_only_tool_needs_no_permission(manager: MCPManager) -> None:
    search = next(t for t in manager.tools if t.name == "docs__search_docs")
    assert search.permission_request({"query": "x"}) is None
    search.trust = "ask"
    assert search.permission_request({"query": "x"}) is not None


def test_call_returns_text_and_citations(manager: MCPManager) -> None:
    search = next(t for t in manager.tools if t.name == "docs__search_docs")
    result = search.run({"query": "Store.watch callback cancel"})
    assert not result.is_error and "watch" in result.content
    assert result.citations[0].label() == "api-reference.md § API reference > Store.watch (v3.2)"


def test_agent_uses_mcp_tool_like_any_other(manager: MCPManager) -> None:
    provider = FakeProvider(
        [[ToolCall("1", "docs__search_docs", {"query": "scan prefix reverse"})], "Use scan."]
    )
    agent = Agent(provider, manager.tools, Permissions(), "sys")
    result = agent.run("how do I iterate keys?")
    assert result.text == "Use scan."
    assert any(c.section == "API reference > Store.scan" for c in result.citations)


def test_bad_server_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    m = MCPManager()
    m.connect("broken", StdioServerParameters(command="/nonexistent/golu-test"), "ask")
    assert m.tools == [] and "broken" in m.warnings[0]
    m.close()


def test_config_loading(tmp_path: Path) -> None:
    (tmp_path / "golu.toml").write_text(
        '[[mcp_servers]]\nname = "docs"\ncommand = "uv"\nargs = ["run", "x"]\ntrust = "read_only"\n'
    )
    [cfg] = load_servers(tmp_path)
    assert (cfg.name, cfg.command, cfg.args, cfg.trust) == ("docs", "uv", ["run", "x"], "read_only")


def test_config_missing_file_and_errors(tmp_path: Path) -> None:
    assert load_servers(tmp_path) == []
    (tmp_path / "golu.toml").write_text(
        '[[mcp_servers]]\nname = "x"\ntrust = "yolo"\ncommand="a"\n'
    )
    with pytest.raises(ConfigError, match="trust"):
        load_servers(tmp_path)
    (tmp_path / "golu.toml").write_text("not = [valid")
    with pytest.raises(ConfigError):
        load_servers(tmp_path)
