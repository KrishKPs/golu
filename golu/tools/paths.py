"""Confines file access to the directory Golu was launched in."""

from pathlib import Path

from golu.tools.base import ToolError


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        """Turn a model-supplied path into an absolute path inside the root.

        `resolve()` follows symlinks and `..`, so a link pointing outside the
        root is rejected too.
        """
        if "\x00" in path:
            raise ToolError("Path contains a NUL byte.")
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(
                f"Path '{path}' is outside the project directory ({self.root}). "
                "Only files inside it can be accessed."
            )
        return target

    def display(self, path: Path) -> str:
        """Path relative to the root, for messages."""
        rel = path.relative_to(self.root).as_posix()
        return rel or "."
