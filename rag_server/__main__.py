"""Run the docs MCP server over stdio: `uv run python -m rag_server`."""

from rag_server.server import build_server

build_server().run("stdio")
