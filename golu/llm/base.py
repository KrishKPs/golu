"""The provider interface.

Every model provider (Anthropic, a fake for tests, others later) implements
`LLMProvider`. The agent and CLI depend only on the types in this file, so
swapping the model never touches the rest of the code.

A conversation is a list of `Message`s. Each message holds content blocks:
plain text, a tool call the model made, or the result we sent back.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolCall:
    """The model asking us to run a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    """Our answer to a `ToolCall`, matched by `tool_call_id`."""

    tool_call_id: str
    content: str
    is_error: bool = False


Block = TextBlock | ToolCall | ToolResultBlock


@dataclass
class Message:
    role: Role
    blocks: list[Block]
    # Provider-native content, replayed unchanged on later requests. Providers use
    # this for things the generic blocks can't represent (e.g. signed thinking
    # blocks). Other code must treat it as opaque. Must be JSON-serializable.
    raw: Any = None

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", [TextBlock(text)])

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.blocks if isinstance(b, ToolCall)]


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told about a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "pause_turn", "other"]


@dataclass
class ModelResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)


class LLMError(Exception):
    """A provider failed in a way the user should see (bad key, network, rate limit)."""


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Send the conversation and return the model's next message.

        `on_text` receives reply text as it streams in, if the provider streams.

        Raises:
            LLMError: if the model can't produce a reply.
        """
