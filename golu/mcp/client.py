"""Keeps MCP server connections open and turns their tools into Golu `Tool`s.

The MCP SDK is async, but the agent loop is plain synchronous code. So the
connections live on a background thread running its own asyncio event loop,
and tool calls are handed to that loop and waited on.

A server that fails to start produces a warning, not a crash: Golu still
works with its built-in tools.
"""

import asyncio
import json
import re
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import IO, Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import MCPServer
from mcp.types import CallToolResult
from mcp.types import Tool as MCPToolInfo

from golu.mcp.config import ServerConfig, Trust
from golu.tools.base import Citation, PermissionRequest, Tool, ToolError, ToolResult

STARTUP_TIMEOUT = 60.0
CALL_TIMEOUT = 120.0


class MCPManager:
    def __init__(self, log_dir: Path | None = None) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._clients: dict[str, Client] = {}
        self._stop: asyncio.Event | None = None
        self._tasks: list[Future[None]] = []
        self._log_dir = log_dir
        self.tools: list[Tool] = []
        self.warnings: list[str] = []

    def connect_all(self, configs: list[ServerConfig], project_root: Path) -> None:
        for cfg in configs:
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env or None,
                cwd=cfg.cwd or str(project_root),
            )
            self.connect(cfg.name, params, cfg.trust)

    def connect(self, name: str, target: StdioServerParameters | MCPServer, trust: Trust) -> None:
        """Start a server (a subprocess, or an in-memory server in tests) and load its tools."""
        ready: Future[list[MCPToolInfo]] = Future()
        task = asyncio.run_coroutine_threadsafe(self._serve(name, target, ready), self._loop)
        self._tasks.append(task)
        try:
            infos = ready.result(timeout=STARTUP_TIMEOUT)
        except Exception as e:
            task.cancel()
            self.warnings.append(f"MCP server '{name}' failed to start: {_describe(e)}")
            return
        self.tools.extend(MCPTool(self, name, info, trust) for info in infos)

    async def _serve(
        self, name: str, target: StdioServerParameters | MCPServer, ready: Future[Any]
    ) -> None:
        # The connection is opened and closed inside this one task, which the
        # SDK requires. The task then waits until close() is called.
        if self._stop is None:
            self._stop = asyncio.Event()
        errlog = self._errlog(name)
        try:
            transport = (
                stdio_client(target, errlog=errlog)
                if isinstance(target, StdioServerParameters)
                else target
            )
            async with Client(transport) as client:
                infos: list[MCPToolInfo] = []
                cursor = None
                while True:
                    page = await client.list_tools(cursor=cursor)
                    infos.extend(page.tools)
                    cursor = page.next_cursor
                    if not cursor:
                        break
                self._clients[name] = client
                ready.set_result(infos)
                await self._stop.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
            if not isinstance(e, Exception):
                raise
        finally:
            self._clients.pop(name, None)
            if errlog is not None:
                errlog.close()

    def _errlog(self, name: str) -> IO[str] | None:
        # A server's stderr goes to a log file instead of cluttering the terminal.
        if self._log_dir is None:
            return open("/dev/null", "w")  # noqa: SIM115 - closed in _serve
        self._log_dir.mkdir(parents=True, exist_ok=True)
        return open(self._log_dir / f"mcp-{name}.log", "w")  # noqa: SIM115

    def call(self, server: str, tool: str, args: dict[str, Any]) -> CallToolResult:
        client = self._clients.get(server)
        if client is None:
            raise ToolError(f"MCP server '{server}' is not connected.")
        future = asyncio.run_coroutine_threadsafe(client.call_tool(tool, args), self._loop)
        try:
            return future.result(timeout=CALL_TIMEOUT)
        except TimeoutError as e:
            future.cancel()
            raise ToolError(f"{server}.{tool} timed out after {CALL_TIMEOUT:.0f}s.") from e
        except Exception as e:
            raise ToolError(f"{server}.{tool} failed: {_describe(e)}") from e

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        for task in self._tasks:
            try:
                task.result(timeout=5)
            except Exception:
                pass  # shutting down: nothing useful to do with errors here
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class MCPTool(Tool):
    """One tool from an MCP server, looking like any other Tool to the agent loop."""

    def __init__(self, manager: MCPManager, server: str, info: MCPToolInfo, trust: Trust) -> None:
        self.manager = manager
        self.server = server
        self.remote_name = info.name
        # Anthropic tool names allow [a-zA-Z0-9_-], max 64 chars.
        self.name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server}__{info.name}")[:64]
        self.description = (info.description or info.name).strip() + f"\n(MCP server: {server})"
        self.input_schema = info.input_schema or {"type": "object", "properties": {}}
        hints = info.annotations
        self.read_only = bool(hints and hints.read_only_hint)
        self.trust = trust

    def permission_request(self, args: dict[str, Any]) -> PermissionRequest | None:
        if self.trust == "all" or (self.trust == "read_only" and self.read_only):
            return None
        return PermissionRequest(
            "external",
            self.name,
            f"Call {self.server}.{self.remote_name}",
            json.dumps(args, indent=2),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        result = self.manager.call(self.server, self.remote_name, args)
        text = _content_text(result)
        if result.is_error:
            return ToolResult(text or "The MCP tool reported an error.", is_error=True)
        return ToolResult(text, citations=_citations(result.structured_content))


def _content_text(result: CallToolResult) -> str:
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else f"[{block.type} content omitted]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, indent=2))
    return "\n".join(parts)


def _citations(structured: Any) -> tuple[Citation, ...]:
    """Any MCP tool whose structured output has `results: [{source, section, ...}]`
    (like our docs server) gets citations recorded for the user to see."""
    if not isinstance(structured, dict):
        return ()
    out = []
    for item in structured.get("results") or []:
        if isinstance(item, dict) and item.get("source") and item.get("section"):
            version = item.get("version")
            out.append(
                Citation(
                    str(item["source"]), str(item["section"]), str(version) if version else None
                )
            )
    return tuple(out)


def _describe(e: BaseException) -> str:
    if isinstance(e, BaseExceptionGroup):
        return "; ".join(_describe(x) for x in e.exceptions)
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
