"""Claude, via the official Anthropic SDK.

Credentials come from the environment only (ANTHROPIC_API_KEY, or an
`ant auth login` profile). The SDK reads them itself; Golu never sees the key.

Two API rules shape this file:
- The model's thinking blocks must be sent back unchanged on later requests,
  so each assistant message keeps the raw response content in `Message.raw`.
- History must be append-only (editing earlier turns invalidates those thinking
  blocks), so old tool output is trimmed by the API's server-side context
  editing rather than by rewriting our message list.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic

from golu.llm.base import (
    Block,
    LLMError,
    LLMProvider,
    Message,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolSpec,
    Usage,
)

DEFAULT_MODEL = "claude-opus-5-5"
_KNOWN_STOPS = {"end_turn", "tool_use", "max_tokens", "refusal", "pause_turn"}


@dataclass(frozen=True)
class ContextEditing:
    """Server-side clearing of old tool results once the prompt gets large."""

    trigger_tokens: int = 150_000
    keep_tool_uses: int = 8
    # Tool results never cleared (e.g. retrieved docs that answers cite).
    exclude_tools: tuple[str, ...] = ()


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 64_000,
        context_editing: ContextEditing | None = None,
    ) -> None:
        self._client = anthropic.Anthropic()
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._editing = context_editing or ContextEditing()

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        params = self._params(messages, system, tools or [])
        for attempt in range(3):
            try:
                with self._client.beta.messages.stream(**params) as stream:
                    for event in stream:
                        if event.type == "text" and on_text:
                            on_text(event.text)
                    response = stream.get_final_message()
                return _to_response(response)
            except ValueError:
                # The streamed tool input was JSON the SDK couldn't parse at all.
                # No tool_use block exists yet to answer, so re-issue the turn.
                if attempt == 2:
                    raise LLMError("Model produced unparseable tool input three times.") from None
            except anthropic.AuthenticationError as e:
                raise LLMError("Authentication failed. Check ANTHROPIC_API_KEY.") from e
            except TypeError as e:
                # The SDK raises TypeError when it finds no credentials at all.
                if "authentication" not in str(e).lower():
                    raise
                raise LLMError("No API credentials found. Set ANTHROPIC_API_KEY.") from e
            except anthropic.RateLimitError as e:
                raise LLMError("Rate limited by the API. Try again shortly.") from e
            except anthropic.APIStatusError as e:
                raise LLMError(f"API error {e.status_code}: {e.message}") from e
            except anthropic.APIConnectionError as e:
                raise LLMError("Could not reach the Anthropic API.") from e
        raise AssertionError("unreachable")

    def _params(
        self, messages: list[Message], system: str | None, tools: list[ToolSpec]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": _to_api_content(m)} for m in messages],
            "output_config": {"effort": self._effort},
            # Server-side fallback: if a safety classifier declines, the API
            # retries on a fallback model inside the same call.
            "fallbacks": "default",
            "betas": ["server-side-fallback-2026-07-01", "context-management-2025-06-27"],
            "context_management": {"edits": [self._clear_edit()]},
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    # Stream large tool inputs (file contents) as they're generated.
                    # The loop validates every input against its schema before running.
                    "eager_input_streaming": True,
                }
                for t in tools
            ]
        return params

    def _clear_edit(self) -> dict[str, Any]:
        edit: dict[str, Any] = {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": self._editing.trigger_tokens},
            "keep": {"type": "tool_uses", "value": self._editing.keep_tool_uses},
        }
        if self._editing.exclude_tools:
            edit["exclude_tools"] = list(self._editing.exclude_tools)
        return edit


def _to_api_content(message: Message) -> list[dict[str, Any]]:
    if message.raw is not None:
        return message.raw
    return [_block_to_api(b) for b in message.blocks]


def _block_to_api(block: Block) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCall):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments}
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_call_id,
        "content": block.content,
        "is_error": block.is_error,
    }


def _to_response(response: Any) -> ModelResponse:
    blocks: list[Block] = []
    for b in response.content:
        if b.type == "text":
            blocks.append(TextBlock(b.text))
        elif b.type == "tool_use":
            args = b.input if isinstance(b.input, dict) else {}
            blocks.append(ToolCall(b.id, b.name, args))
    raw = [b.model_dump(mode="json", exclude_unset=True) for b in response.content]
    stop = response.stop_reason
    stop_reason: StopReason = stop if stop in _KNOWN_STOPS else "other"
    usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
    return ModelResponse(Message("assistant", blocks, raw=raw), stop_reason, usage)
