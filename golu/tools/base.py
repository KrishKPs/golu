"""The Tool interface. The agent loop only knows about this, so built-in tools
and MCP tools look identical to it.

A tool either returns a `ToolResult` or raises `ToolError` with a message
written for the model; the loop turns that into an error result so the model
can read it and recover.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from golu.llm.base import ToolSpec


class ToolError(Exception):
    """A failure the model should see and can recover from (bad path, no match, ...)."""


@dataclass(frozen=True)
class Citation:
    """Where a piece of retrieved documentation came from."""

    source: str
    section: str
    version: str | None = None

    def label(self) -> str:
        version = f" (v{self.version})" if self.version else ""
        return f"{self.source} § {self.section}{version}"


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    citations: tuple[Citation, ...] = ()


PermissionKind = Literal["edit", "command", "external"]


@dataclass(frozen=True)
class PermissionRequest:
    """What a tool wants to do, shown to the user before it happens."""

    kind: PermissionKind
    tool: str
    summary: str  # one line, e.g. "Edit golu/cli.py"
    detail: str = ""  # the diff, or the exact command
    command: str | None = None  # set for kind="command"; checked against the denylist


class Tool(ABC):
    name: str
    description: str  # written for the model: what it does and when to use it
    input_schema: dict[str, Any]

    def permission_request(self, args: dict[str, Any]) -> PermissionRequest | None:
        """Describe the side effect for the user to approve, or None for read-only tools.

        May raise ToolError if the call is invalid (so we don't ask about a doomed edit).
        """
        return None

    @abstractmethod
    def run(self, args: dict[str, Any]) -> ToolResult:
        """Do the work. Raise ToolError for failures the model should see."""

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)
