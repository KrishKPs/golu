"""Tools that change files: write_file and edit_file. Each asks permission with a diff."""

from typing import Any

from golu.diffs import unified_diff
from golu.tools.base import PermissionRequest, Tool, ToolError, ToolResult
from golu.tools.files import read_text
from golu.tools.paths import Workspace


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Create a new file, or replace a file's entire contents. Creates parent folders. "
        "Prefer edit_file for changing part of an existing file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the project root."},
            "content": {"type": "string", "description": "The complete new file contents."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def permission_request(self, args: dict[str, Any]) -> PermissionRequest:
        path = self.ws.resolve(args["path"])
        if path.is_dir():
            raise ToolError(f"{args['path']} is a directory.")
        old = read_text(path, args["path"]) if path.exists() else ""
        verb = "Overwrite" if path.exists() else "Create"
        shown = self.ws.display(path)
        return PermissionRequest(
            "edit", self.name, f"{verb} {shown}", unified_diff(old, args["content"], shown)
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        path = self.ws.resolve(args["path"])
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        lines = args["content"].count("\n") + (0 if args["content"].endswith("\n") else 1)
        return ToolResult(f"{'Updated' if existed else 'Created'} {args['path']} ({lines} lines).")


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact piece of text in an existing file. 'old_string' must match the "
        "file exactly (including indentation) and appear exactly once, unless replace_all "
        "is true. Include enough surrounding lines to make it unique. Read the file first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the project root."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def _apply(self, args: dict[str, Any]) -> tuple[str, str]:
        """Return (old file text, new file text), or raise ToolError."""
        path = self.ws.resolve(args["path"])
        old = read_text(path, args["path"])
        target, replacement = args["old_string"], args["new_string"]
        if target == "":
            raise ToolError("old_string is empty. Use write_file to create a file.")
        if target == replacement:
            raise ToolError("old_string and new_string are identical; nothing to change.")
        count = old.count(target)
        if count == 0:
            raise ToolError(
                f"old_string was not found in {args['path']}. Read the file and copy the "
                "text exactly, including whitespace."
            )
        if count > 1 and not args.get("replace_all"):
            raise ToolError(
                f"old_string appears {count} times in {args['path']}. Add surrounding lines "
                "to make it unique, or set replace_all to true."
            )
        return old, old.replace(target, replacement)

    def permission_request(self, args: dict[str, Any]) -> PermissionRequest:
        old, new = self._apply(args)
        shown = self.ws.display(self.ws.resolve(args["path"]))
        return PermissionRequest("edit", self.name, f"Edit {shown}", unified_diff(old, new, shown))

    def run(self, args: dict[str, Any]) -> ToolResult:
        old, new = self._apply(args)
        self.ws.resolve(args["path"]).write_text(new, encoding="utf-8")
        replaced = old.count(args["old_string"]) if args.get("replace_all") else 1
        return ToolResult(f"Edited {args['path']} ({replaced} replacement(s)).")
