"""Checks request building and response parsing. No network calls."""

from types import SimpleNamespace

import pytest

from golu.llm.anthropic import AnthropicProvider, ContextEditing, _to_response
from golu.llm.base import Message, TextBlock, ToolCall, ToolResultBlock, ToolSpec


class Block(SimpleNamespace):
    def model_dump(self, **_: object) -> dict:
        return dict(vars(self))


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> AnthropicProvider:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    return AnthropicProvider(context_editing=ContextEditing(exclude_tools=("docs__search_docs",)))


def test_params(provider: AnthropicProvider) -> None:
    messages = [
        Message.user("hi"),
        Message(
            "assistant",
            [ToolCall("t1", "read_file", {"path": "a"})],
            raw=[
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
            ],
        ),
        Message("user", [ToolResultBlock("t1", "data", False)]),
    ]
    params = provider._params(messages, "sys", [ToolSpec("read_file", "d", {"type": "object"})])
    assert params["model"] == "claude-opus-5-5"
    assert params["system"] == "sys"
    assert params["messages"][1]["content"][0]["type"] == "thinking"  # raw replayed unchanged
    assert params["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "data", "is_error": False}
    ]
    assert params["tools"][0]["eager_input_streaming"] is True
    edit = params["context_management"]["edits"][0]
    assert edit["type"] == "clear_tool_uses_20250919"
    assert edit["exclude_tools"] == ["docs__search_docs"]
    assert "thinking" not in params  # Opus 5.5 thinks adaptively by default


def test_to_response() -> None:
    response = SimpleNamespace(
        content=[
            Block(type="thinking", thinking="", signature="sig"),
            Block(type="text", text="Let me look."),
            Block(type="tool_use", id="t1", name="list_dir", input={}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    out = _to_response(response)
    assert out.stop_reason == "tool_use"
    assert out.message.blocks == [TextBlock("Let me look."), ToolCall("t1", "list_dir", {})]
    assert out.message.raw[0]["signature"] == "sig"
    assert out.usage.input_tokens == 10


def test_unknown_stop_reason_mapped() -> None:
    response = SimpleNamespace(
        content=[],
        stop_reason="something_new",
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    assert _to_response(response).stop_reason == "other"
