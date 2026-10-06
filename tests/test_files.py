from pathlib import Path

import pytest

from golu.tools.base import ToolError
from golu.tools.files import ListDir, ReadFile, Search
from golu.tools.paths import Workspace


def test_read_file_numbers_lines(ws: Workspace) -> None:
    out = ReadFile(ws).run({"path": "src/app.py"}).content
    assert "     1\tdef add(a, b):" in out
    assert "     2\t    return a + b" in out


def test_read_file_offset_and_limit(ws: Workspace) -> None:
    (ws.root / "big.txt").write_text("\n".join(f"line {i}" for i in range(1, 101)))
    out = ReadFile(ws).run({"path": "big.txt", "offset": 10, "limit": 2}).content
    assert "line 10" in out and "line 11" in out and "line 12" not in out
    assert "offset=12" in out


def test_read_file_missing(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="not found"):
        ReadFile(ws).run({"path": "nope.py"})


def test_read_file_binary(ws: Workspace) -> None:
    (ws.root / "x.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        ReadFile(ws).run({"path": "x.bin"})


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "src/../../x"])
def test_paths_outside_root_rejected(ws: Workspace, path: str) -> None:
    with pytest.raises(ToolError, match="outside the project"):
        ReadFile(ws).run({"path": path})


def test_symlink_escape_rejected(ws: Workspace, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("secret")
    (ws.root / "link.txt").symlink_to(outside)
    with pytest.raises(ToolError, match="outside the project"):
        ReadFile(ws).run({"path": "link.txt"})


def test_list_dir(ws: Workspace) -> None:
    out = ListDir(ws).run({}).content
    assert out.splitlines() == ["src/", "README.md"]


def test_list_dir_not_a_directory(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="Not a directory"):
        ListDir(ws).run({"path": "README.md"})


def test_search_finds_matches(ws: Workspace) -> None:
    out = Search(ws).run({"pattern": r"return \w"}).content
    assert out == "src/app.py:2: return a + b"


def test_search_glob_and_skip_dirs(ws: Workspace) -> None:
    (ws.root / ".venv").mkdir()
    (ws.root / ".venv" / "lib.py").write_text("return x\n")
    assert Search(ws).run({"pattern": "return", "glob": "*.md"}).content == "No matches."
    assert ".venv" not in Search(ws).run({"pattern": "return"}).content


def test_search_bad_regex(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="Invalid regular expression"):
        Search(ws).run({"pattern": "("})


def test_workspace_display(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    assert ws.display(tmp_path / "a" / "b.py") == "a/b.py"
