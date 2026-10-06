from golu.agent.loop import Agent
from golu.llm.base import Message, ModelResponse, TextBlock, ToolCall, ToolResultBlock
from golu.llm.fake import FakeProvider
from golu.permissions import Permissions
from golu.tools import builtin_tools
from golu.tools.paths import Workspace


def make_agent(ws: Workspace, replies: list, mode: str = "auto", prompter=None) -> Agent:
    provider = FakeProvider(replies)
    return Agent(provider, builtin_tools(ws), Permissions(mode=mode, prompter=prompter), "sys")


def last_results(agent: Agent, index: int = -2) -> list[ToolResultBlock]:
    return agent.conversation.messages[index].blocks  # type: ignore[return-value]


def test_loops_until_no_tool_calls(ws: Workspace) -> None:
    agent = make_agent(ws, [[ToolCall("1", "read_file", {"path": "README.md"})], "It's a demo."])
    result = agent.run("what is this?")
    assert result.text == "It's a demo."
    roles = [m.role for m in agent.conversation.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    tool_result = agent.conversation.messages[2].blocks[0]
    assert isinstance(tool_result, ToolResultBlock) and "# Demo" in tool_result.content


def test_parallel_calls_answered_in_one_message(ws: Workspace) -> None:
    calls = [ToolCall("a", "list_dir", {}), ToolCall("b", "read_file", {"path": "README.md"})]
    agent = make_agent(ws, [calls, "done"])
    agent.run("go")
    results = agent.conversation.messages[2].blocks
    assert [r.tool_call_id for r in results] == ["a", "b"]  # type: ignore[union-attr]


def test_tool_errors_go_back_to_model(ws: Workspace) -> None:
    agent = make_agent(
        ws,
        [
            [
                ToolCall("1", "read_file", {"path": "missing.py"}),
                ToolCall("2", "nope", {}),
                ToolCall("3", "read_file", {}),
            ],
            "sorry",
        ],
    )
    agent.run("go")
    r1, r2, r3 = agent.conversation.messages[2].blocks
    assert r1.is_error and "not found" in r1.content  # type: ignore[union-attr]
    assert r2.is_error and "Unknown tool" in r2.content  # type: ignore[union-attr]
    assert r3.is_error and "Missing required" in r3.content  # type: ignore[union-attr]


def test_denied_edit_is_not_applied(ws: Workspace) -> None:
    edit = ToolCall("1", "write_file", {"path": "x.py", "content": "boom"})
    agent = make_agent(ws, [[edit], "ok"], mode="ask", prompter=lambda req: "no")
    agent.run("write it")
    result = agent.conversation.messages[2].blocks[0]
    assert result.is_error and "declined" in result.content  # type: ignore[union-attr]
    assert not (ws.root / "x.py").exists()


def test_approved_edit_shows_diff_then_applies(ws: Workspace) -> None:
    seen = []
    edit = ToolCall("1", "edit_file", {"path": "src/app.py", "old_string": "+", "new_string": "*"})
    agent = make_agent(ws, [[edit], "ok"], mode="ask", prompter=lambda r: seen.append(r) or "yes")
    agent.run("multiply")
    assert "+    return a * b" in seen[0].detail
    assert "a * b" in (ws.root / "src" / "app.py").read_text()


def test_denylisted_command_blocked_in_auto_mode(ws: Workspace) -> None:
    agent = make_agent(ws, [[ToolCall("1", "run_command", {"command": "sudo ls"})], "ok"])
    agent.run("go")
    result = agent.conversation.messages[2].blocks[0]
    assert result.is_error and "denylist" in result.content  # type: ignore[union-attr]


def test_long_output_truncated(ws: Workspace) -> None:
    (ws.root / "big.txt").write_text("x" * 100 + "\n" + "y" * 100_000)
    agent = make_agent(ws, [[ToolCall("1", "read_file", {"path": "big.txt"})], "ok"])
    agent.max_tool_output = 1000
    agent.run("go")
    content = agent.conversation.messages[2].blocks[0].content  # type: ignore[union-attr]
    assert "truncated" in content and len(content) < 1200


def test_max_tokens_mid_tool_call_not_run(ws: Workspace) -> None:
    cut = ModelResponse(
        Message("assistant", [ToolCall("1", "write_file", {"path": "x.py", "content": "par"})]),
        "max_tokens",
    )
    agent = make_agent(ws, [cut, "ok"])
    agent.run("go")
    assert not (ws.root / "x.py").exists()
    assert agent.conversation.messages[2].blocks[0].is_error  # type: ignore[union-attr]


def test_stops_after_max_steps(ws: Workspace) -> None:
    agent = make_agent(ws, [[ToolCall(str(i), "list_dir", {})] for i in range(3)])
    agent.max_steps = 3
    assert agent.run("loop forever").stop_reason == "max_steps"


def test_refusal_returns(ws: Workspace) -> None:
    refusal = ModelResponse(Message("assistant", [TextBlock("")]), "refusal")
    assert make_agent(ws, [refusal]).run("x").stop_reason == "refusal"
