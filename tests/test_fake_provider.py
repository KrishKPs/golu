import pytest

from golu.llm.base import LLMError, Message, ToolCall
from golu.llm.fake import FakeProvider


def test_replays_in_order_and_records_calls() -> None:
    p = FakeProvider(["a", [ToolCall("1", "list_dir", {})]])
    first = p.complete([Message.user("x")], system="sys")
    assert first.message.text == "a" and first.stop_reason == "end_turn"
    second = p.complete([Message.user("y")])
    assert second.stop_reason == "tool_use" and second.message.tool_calls[0].name == "list_dir"
    assert p.calls[0]["system"] == "sys"


def test_streams_text_to_callback() -> None:
    seen: list[str] = []
    FakeProvider(["hello"]).complete([Message.user("x")], on_text=seen.append)
    assert seen == ["hello"]


def test_raises_when_exhausted() -> None:
    with pytest.raises(LLMError):
        FakeProvider([]).complete([Message.user("x")])
