from pathlib import Path

from golu.agent import session as sessions
from golu.agent.context import Conversation, truncate_output
from golu.llm.base import Message, TextBlock, ToolCall, ToolResultBlock


def test_roundtrip_keeps_blocks_and_raw(tmp_path: Path) -> None:
    s = sessions.Session.new("sys", ["read_file"])
    s.conversation.append(Message.user("hi"))
    s.conversation.append(
        Message(
            "assistant",
            [TextBlock("t"), ToolCall("1", "read_file", {"path": "a"})],
            raw=[{"type": "thinking", "thinking": "", "signature": "abc"}],
        )
    )
    s.conversation.append(Message("user", [ToolResultBlock("1", "content", True)]))
    sessions.save(tmp_path, s)
    loaded = sessions.load(tmp_path, s.id)
    assert loaded.conversation.messages == s.conversation.messages
    assert sessions.latest_id(tmp_path) == s.id


def test_adapt_drops_raw_when_setup_changes() -> None:
    s = sessions.Session.new("sys", ["a"])
    s.conversation.append(Message("assistant", [TextBlock("x")], raw=[{"type": "thinking"}]))
    assert not s.adapt_to("sys", ["a"])
    assert s.conversation.messages[0].raw is not None
    assert s.adapt_to("sys", ["a", "docs__search_docs"])
    assert s.conversation.messages[0].raw is None


def test_latest_none(tmp_path: Path) -> None:
    assert sessions.latest_id(tmp_path) is None


def test_truncate_output_keeps_head_and_tail() -> None:
    text = "HEAD" + "x" * 10_000 + "TAIL"
    out = truncate_output(text, 1000)
    assert out.startswith("HEAD") and out.endswith("TAIL") and "truncated" in out
    assert truncate_output("short", 1000) == "short"


def test_conversation_usage() -> None:
    from golu.llm.base import Usage

    c = Conversation()
    c.record_usage(Usage(100, 10))
    c.record_usage(Usage(150, 5))
    assert c.context_tokens == 150 and c.total_output_tokens == 15
