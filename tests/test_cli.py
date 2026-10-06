import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from golu import cli
from golu.llm.base import ToolCall
from golu.llm.fake import FakeProvider

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "hello.py").write_text("print('hi')\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def use_fake() -> Iterator[list[FakeProvider]]:
    """Each CLI run gets the next provider from the list."""
    providers: list[FakeProvider] = []
    cli.provider_factory = lambda *_: providers.pop(0)
    yield providers
    cli.provider_factory = None


def test_one_shot_prints_reply_and_saves_session(project: Path, use_fake: list) -> None:
    fake = FakeProvider(["Hello from the model."])
    use_fake.append(fake)
    result = runner.invoke(cli.app, ["say hi"])
    assert result.exit_code == 0, result.output
    assert "Hello from the model." in result.output
    assert fake.calls[0]["messages"][0].text == "say hi"
    assert "Project root" in fake.calls[0]["system"]
    [saved] = (project / ".golu" / "sessions").glob("*.json")
    assert len(json.loads(saved.read_text())["messages"]) == 2


def test_error_exits_nonzero(project: Path, use_fake: list) -> None:
    use_fake.append(FakeProvider([]))
    result = runner.invoke(cli.app, ["say hi"])
    assert result.exit_code == 1 and "Error" in result.output


def test_edit_asks_and_shows_diff(project: Path, use_fake: list) -> None:
    edit = ToolCall("1", "edit_file", {"path": "hello.py", "old_string": "hi", "new_string": "yo"})
    use_fake.append(FakeProvider([[edit], "Done."]))
    result = runner.invoke(cli.app, ["change it"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Edit hello.py" in result.output and "+print('yo')" in result.output
    assert (project / "hello.py").read_text() == "print('yo')\n"


def test_declined_edit_not_applied(project: Path, use_fake: list) -> None:
    edit = ToolCall("1", "write_file", {"path": "new.py", "content": "x"})
    use_fake.append(FakeProvider([[edit], "OK, I won't."]))
    result = runner.invoke(cli.app, ["write"], input="n\n")
    assert result.exit_code == 0
    assert not (project / "new.py").exists()


def test_yes_flag_auto_approves(project: Path, use_fake: list) -> None:
    edit = ToolCall("1", "write_file", {"path": "new.py", "content": "x"})
    use_fake.append(FakeProvider([[edit], "Created."]))
    assert runner.invoke(cli.app, ["write", "--yes"]).exit_code == 0
    assert (project / "new.py").read_text() == "x"


def test_interactive_keeps_history(project: Path, use_fake: list) -> None:
    fake = FakeProvider(["first", "second"])
    use_fake.append(fake)
    result = runner.invoke(cli.app, [], input="one\ntwo\nexit\n")
    assert result.exit_code == 0
    texts = [(m.role, m.text) for m in fake.calls[1]["messages"]]
    assert texts == [("user", "one"), ("assistant", "first"), ("user", "two")]
    assert "--resume" in result.output


def test_continue_resumes_latest_session(project: Path, use_fake: list) -> None:
    use_fake.append(FakeProvider(["remembered"]))
    runner.invoke(cli.app, ["remember 42"])
    fake = FakeProvider(["it was 42"])
    use_fake.append(fake)
    result = runner.invoke(cli.app, ["what number?", "--continue"])
    assert result.exit_code == 0 and "Resumed session" in result.output
    assert [m.text for m in fake.calls[0]["messages"]] == [
        "remember 42",
        "remembered",
        "what number?",
    ]


def test_bad_resume_id(project: Path, use_fake: list) -> None:
    result = runner.invoke(cli.app, ["x", "--resume", "nope"])
    assert result.exit_code == 2 and "No saved session" in result.output


def test_mcp_config_error_reported(project: Path, use_fake: list) -> None:
    (project / "golu.toml").write_text("[[mcp_servers]]\nname = 'x'\n")
    result = runner.invoke(cli.app, ["hi"])
    assert result.exit_code == 2 and "Config error" in result.output


def test_broken_mcp_server_is_warning(project: Path, use_fake: list) -> None:
    (project / "golu.toml").write_text(
        "[[mcp_servers]]\nname = 'docs'\ncommand = '/nonexistent/server'\n"
    )
    use_fake.append(FakeProvider(["still works"]))
    result = runner.invoke(cli.app, ["hi"])
    assert result.exit_code == 0
    assert "failed to start" in result.output and "still works" in result.output
