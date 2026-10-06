"""Conversation history, output truncation, and token tracking.

The history is append-only on purpose: the model's earlier thinking is only
valid if every earlier message is sent back byte-for-byte unchanged. So we
never rewrite old messages. Instead:
- tool output is truncated *before* it enters the history, and
- old tool results are cleared server-side once the prompt gets large
  (configured in the provider).
"""

from dataclasses import dataclass, field
from typing import Any

from golu.llm.base import Block, Message, TextBlock, ToolCall, ToolResultBlock, Usage

DEFAULT_MAX_TOOL_OUTPUT = 30_000  # characters, roughly 7-8k tokens


def truncate_output(text: str, limit: int = DEFAULT_MAX_TOOL_OUTPUT) -> str:
    """Keep the start and end of long output, which is where the useful parts usually are."""
    if len(text) <= limit:
        return text
    head = text[: limit * 2 // 3]
    tail = text[-(limit // 3) :]
    cut = len(text) - len(head) - len(tail)
    return f"{head}\n\n[... {cut} characters truncated ...]\n\n{tail}"


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    last_usage: Usage = field(default_factory=Usage)
    total_output_tokens: int = 0

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def record_usage(self, usage: Usage) -> None:
        self.last_usage = usage
        self.total_output_tokens += usage.output_tokens

    @property
    def context_tokens(self) -> int:
        """Size of the most recent prompt, as reported by the provider."""
        return self.last_usage.input_tokens

    # --- Serialization (for session resume) ---

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [_message_to_dict(m) for m in self.messages]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        return cls([_message_from_dict(m) for m in data.get("messages", [])])


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {"role": m.role, "blocks": [_block_to_dict(b) for b in m.blocks], "raw": m.raw}


def _block_to_dict(b: Block) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolCall):
        return {"type": "tool_call", "id": b.id, "name": b.name, "arguments": b.arguments}
    return {"type": "tool_result", "id": b.tool_call_id, "content": b.content, "error": b.is_error}


def _message_from_dict(d: dict[str, Any]) -> Message:
    blocks: list[Block] = []
    for b in d["blocks"]:
        if b["type"] == "text":
            blocks.append(TextBlock(b["text"]))
        elif b["type"] == "tool_call":
            blocks.append(ToolCall(b["id"], b["name"], b["arguments"]))
        else:
            blocks.append(ToolResultBlock(b["id"], b["content"], b["error"]))
    return Message(d["role"], blocks, raw=d.get("raw"))
