import pytest

from golu.tools.base import ToolError
from golu.tools.edit import EditFile, WriteFile
from golu.tools.paths import Workspace


def test_write_new_file_and_diff(ws: Workspace) -> None:
    tool = WriteFile(ws)
    args = {"path": "pkg/new.py", "content": "x = 1\n"}
    req = tool.permission_request(args)
    assert req.kind == "edit" and req.summary == "Create pkg/new.py"
    assert "+x = 1" in req.detail
    assert "Created" in tool.run(args).content
    assert (ws.root / "pkg" / "new.py").read_text() == "x = 1\n"


def test_write_outside_root_rejected(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="outside"):
        WriteFile(ws).permission_request({"path": "../evil.py", "content": ""})


def test_edit_replaces_unique_text(ws: Workspace) -> None:
    tool = EditFile(ws)
    args = {"path": "src/app.py", "old_string": "a + b", "new_string": "a - b"}
    req = tool.permission_request(args)
    assert "-    return a + b" in req.detail and "+    return a - b" in req.detail
    tool.run(args)
    assert "return a - b" in (ws.root / "src" / "app.py").read_text()


def test_edit_not_found(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="not found"):
        EditFile(ws).permission_request(
            {"path": "src/app.py", "old_string": "zzz", "new_string": "y"}
        )


def test_edit_ambiguous_needs_replace_all(ws: Workspace) -> None:
    (ws.root / "d.txt").write_text("a\na\n")
    tool = EditFile(ws)
    with pytest.raises(ToolError, match="appears 2 times"):
        tool.run({"path": "d.txt", "old_string": "a", "new_string": "b"})
    tool.run({"path": "d.txt", "old_string": "a", "new_string": "b", "replace_all": True})
    assert (ws.root / "d.txt").read_text() == "b\nb\n"


def test_edit_missing_file(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="not found"):
        EditFile(ws).run({"path": "nope.py", "old_string": "a", "new_string": "b"})
