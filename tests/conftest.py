from pathlib import Path

import pytest

from golu.tools.paths import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    return Workspace(tmp_path)
