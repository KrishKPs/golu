"""Read-only file tools: read_file, list_dir, search. These never need approval."""

import re
from pathlib import Path
from typing import Any

from golu.tools.base import Tool, ToolError, ToolResult
from golu.tools.paths import Workspace

# Directories that are never useful to list or search.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".golu",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    "dist",
    "build",
}


def read_text(path: Path, shown: str) -> str:
    if not path.exists():
        raise ToolError(f"File not found: {shown}")
    if path.is_dir():
        raise ToolError(f"{shown} is a directory. Use list_dir instead.")
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ToolError(f"{shown} looks like a binary file; it can't be read as text.")
    return data.decode("utf-8", errors="replace")


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file in the project. Returns lines prefixed with line numbers. "
        "For large files, use offset and limit to read a section. "
        "Always read a file before editing it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the project root."},
            "offset": {"type": "integer", "description": "First line to read (1-based)."},
            "limit": {"type": "integer", "description": "Maximum number of lines (default 2000)."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def run(self, args: dict[str, Any]) -> ToolResult:
        path = self.ws.resolve(args["path"])
        lines = read_text(path, args["path"]).splitlines()
        offset = max(1, args.get("offset", 1))
        limit = max(1, args.get("limit", 2000))
        chunk = lines[offset - 1 : offset - 1 + limit]
        if not chunk:
            return ToolResult(f"{args['path']} has {len(lines)} lines; nothing at line {offset}.")
        body = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(chunk, start=offset))
        end = offset + len(chunk) - 1
        if end < len(lines):
            body += f"\n... ({len(lines) - end} more lines; use offset={end + 1} to continue)"
        return ToolResult(body)


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List the files and folders in a project directory. Folders end with '/'. "
        "Use this to explore the project structure."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory relative to the root (default '.').",
            },
        },
        "additionalProperties": False,
    }
    max_entries = 500

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def run(self, args: dict[str, Any]) -> ToolResult:
        shown = args.get("path", ".")
        path = self.ws.resolve(shown)
        if not path.is_dir():
            raise ToolError(f"Not a directory: {shown}")
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        names = [p.name + "/" if p.is_dir() else p.name for p in entries]
        if not names:
            return ToolResult(f"{shown} is empty.")
        extra = len(names) - self.max_entries
        out = "\n".join(names[: self.max_entries])
        if extra > 0:
            out += f"\n... and {extra} more"
        return ToolResult(out)


class Search(Tool):
    name = "search"
    description = (
        "Search file contents in the project with a regular expression (Python syntax). "
        "Returns matching lines as 'path:line: text'. Use 'glob' to limit file types, "
        "e.g. '*.py'. Skips .git, virtualenvs, and node_modules."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Directory or file to search (default '.')."},
            "glob": {"type": "string", "description": "Only search files matching this glob."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    max_matches = 200

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            raise ToolError(f"Invalid regular expression: {e}") from e
        start = self.ws.resolve(args.get("path", "."))
        if not start.exists():
            raise ToolError(f"Path not found: {args.get('path', '.')}")
        matches: list[str] = []
        for file in self._files(start, args.get("glob")):
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable: skip silently
            for n, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{self.ws.display(file)}:{n}: {line.strip()[:300]}")
                    if len(matches) >= self.max_matches:
                        matches.append(
                            f"... stopped at {self.max_matches} matches; narrow the search"
                        )
                        return ToolResult("\n".join(matches))
        return ToolResult("\n".join(matches) if matches else "No matches.")

    def _files(self, start: Path, glob: str | None) -> list[Path]:
        if start.is_file():
            return [start]
        files = []
        for p in sorted(start.rglob(glob or "*")):
            rel_parts = p.relative_to(start).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            # rglob follows the pattern but not our sandbox; re-check symlinks.
            if p.is_file() and p.resolve().is_relative_to(self.ws.root):
                files.append(p)
        return files
