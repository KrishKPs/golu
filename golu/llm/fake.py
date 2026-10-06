"""A provider that replays scripted replies. Used by tests so they never call a real model."""

from collections.abc import Callable
from typing import Any

from golu.llm.base import (
    LLMError,
    LLMProvider,
    Message,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolSpec,
)

# A scripted reply: plain text, a list of blocks, or a full ModelResponse.
Script = str | list[TextBlock | ToolCall] | ModelResponse


class FakeProvider(LLMProvider):
    """Returns `replies` in order, and records every call for assertions."""

    def __init__(self, replies: list[Script]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": list(messages), "system": system, "tools": tools})
        if not self.replies:
            raise LLMError("FakeProvider ran out of scripted replies.")
        reply = self.replies.pop(0)
        if isinstance(reply, ModelResponse):
            return reply
        blocks = [TextBlock(reply)] if isinstance(reply, str) else list(reply)
        message = Message("assistant", blocks)
        if on_text and message.text:
            on_text(message.text)
        return ModelResponse(message, "tool_use" if message.tool_calls else "end_turn")
