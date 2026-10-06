import pytest

from golu.tools.base import ToolError
from golu.tools.paths import Workspace
from golu.tools.shell import RunCommand


def test_runs_in_project_root(ws: Workspace) -> None:
    result = RunCommand(ws).run({"command": "ls src"})
    assert not result.is_error
    assert "exit code: 0" in result.content and "app.py" in result.content


def test_nonzero_exit_is_error(ws: Workspace) -> None:
    result = RunCommand(ws).run({"command": "echo oops >&2; exit 3"})
    assert result.is_error
    assert "exit code: 3" in result.content and "oops" in result.content


def test_timeout_kills_command(ws: Workspace) -> None:
    result = RunCommand(ws).run({"command": "sleep 5", "timeout_seconds": 1})
    assert result.is_error and "timed out after 1s" in result.content


def test_api_keys_hidden(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-1234567890")
    out = RunCommand(ws).run({"command": "env; echo sk-test-1234567890"}).content
    assert "sk-test-1234567890" not in out
    assert "[redacted $ANTHROPIC_API_KEY]" in out


def test_permission_request_shows_command(ws: Workspace) -> None:
    req = RunCommand(ws).permission_request({"command": "pytest -q"})
    assert req.kind == "command" and req.command == "pytest -q"


def test_empty_command_rejected(ws: Workspace) -> None:
    with pytest.raises(ToolError):
        RunCommand(ws).permission_request({"command": "  "})
