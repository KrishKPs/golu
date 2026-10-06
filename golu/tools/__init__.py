"""Built-in tools."""

from golu.tools.base import Tool
from golu.tools.edit import EditFile, WriteFile
from golu.tools.files import ListDir, ReadFile, Search
from golu.tools.paths import Workspace
from golu.tools.shell import RunCommand


def builtin_tools(workspace: Workspace) -> list[Tool]:
    return [
        ReadFile(workspace),
        ListDir(workspace),
        Search(workspace),
        EditFile(workspace),
        WriteFile(workspace),
        RunCommand(workspace),
    ]
