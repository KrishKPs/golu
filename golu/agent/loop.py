"""The agent loop: model -> tool calls -> results -> model, until the model stops.

The loop only knows the `Tool` interface. Each tool call goes through:
validate input -> permission check (if the tool has side effects) -> run.
Any failure becomes an error result the model can read, never a crash.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from golu.agent.context import DEFAULT_MAX_TOOL_OUTPUT, Conversation, truncate_output
from golu.llm.base import LLMProvider, Message, ToolCall, ToolResultBlock
from golu.permissions import Permissions
from golu.secrets import redact
from golu.tools.base import Citation, Tool, ToolError, ToolResult
from golu.tools.schema import validate


@dataclass
class AgentEvents:
    """Hooks for the UI. All optional."""

    on_text: Callable[[str], None] | None = None
    on_tool_start: Callable[[ToolCall], None] | None = None
    on_tool_end: Callable[[ToolCall, ToolResult], None] | None = None
    on_model_start: Callable[[], None] | None = None


@dataclass
class TurnResult:
    text: str  # the model's final reply
    stop_reason: str
    citations: list[Citation] = field(default_factory=list)  # docs retrieved this turn


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool],
        permissions: Permissions,
        system: str,
        conversation: Conversation | None = None,
        events: AgentEvents | None = None,
        max_steps: int = 50,
        max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT,
    ) -> None:
        self.provider = provider
        self.tools = {t.name: t for t in tools}
        self.permissions = permissions
        self.system = system
        self.conversation = conversation or Conversation()
        self.events = events or AgentEvents()
        self.max_steps = max_steps
        self.max_tool_output = max_tool_output
        self._specs = [t.spec() for t in tools]

    def run(self, user_text: str) -> TurnResult:
        """Handle one user message, running tools until the model gives a final reply."""
        self.conversation.append(Message.user(user_text))
        citations: list[Citation] = []
        for _ in range(self.max_steps):
            if self.events.on_model_start:
                self.events.on_model_start()
            response = self.provider.complete(
                self.conversation.messages,
                system=self.system,
                tools=self._specs,
                on_text=self.events.on_text,
            )
            self.conversation.append(response.message)
            self.conversation.record_usage(response.usage)
            calls = response.message.tool_calls

            if response.stop_reason == "refusal":
                return TurnResult(response.message.text, "refusal", citations)
            if response.stop_reason == "pause_turn":
                continue  # a server-side tool paused; resending continues it
            if not calls:
                return TurnResult(response.message.text, response.stop_reason, citations)

            if response.stop_reason == "max_tokens":
                # The reply was cut off mid tool call, so its input may be incomplete.
                # Every call still needs a result, so answer each with an error.
                results = [
                    ToolResultBlock(
                        c.id,
                        "Your reply hit the output limit, so this tool input may be "
                        "incomplete and was not run. Retry with a smaller input "
                        "(e.g. split a large file write into several edits).",
                        is_error=True,
                    )
                    for c in calls
                ]
            else:
                results = []
                try:
                    for call in calls:
                        result = self._execute(call)
                        citations.extend(c for c in result.citations if c not in citations)
                        results.append(ToolResultBlock(call.id, result.content, result.is_error))
                except KeyboardInterrupt:
                    # Every tool call needs a result, or the saved history is invalid.
                    done = {r.tool_call_id for r in results}
                    results += [
                        ToolResultBlock(c.id, "Interrupted by the user.", is_error=True)
                        for c in calls
                        if c.id not in done
                    ]
                    self.conversation.append(Message("user", list(results)))
                    raise
            # All results for one model turn go back together in a single message.
            self.conversation.append(Message("user", list(results)))

        return TurnResult(
            f"Stopped after {self.max_steps} steps without finishing. "
            "Send another message to let me continue.",
            "max_steps",
            citations,
        )

    def _execute(self, call: ToolCall) -> ToolResult:
        if self.events.on_tool_start:
            self.events.on_tool_start(call)
        result = self._run_tool(call)
        # Truncate before the output enters history; redact any API key that leaked in.
        content = redact(truncate_output(result.content, self.max_tool_output))
        result = ToolResult(content, result.is_error, result.citations)
        if self.events.on_tool_end:
            self.events.on_tool_end(call, result)
        return result

    def _run_tool(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return _error(f"Unknown tool '{call.name}'. Available: {', '.join(self.tools)}.")
        problem = validate(tool.input_schema, call.arguments)
        if problem:
            return _error(f"Invalid input for {call.name}: {problem}")
        try:
            request = tool.permission_request(call.arguments)
            if request is not None:
                decision = self.permissions.decide(request)
                if not decision.allowed:
                    return _error(
                        f"Not allowed: {request.summary} was {decision.reason}. "
                        "Do not retry the same action; explain or ask the user instead."
                    )
            return tool.run(call.arguments)
        except ToolError as e:
            return _error(str(e))
        except Exception as e:  # a bug in a tool must not crash the session
            return _error(f"{call.name} failed unexpectedly: {type(e).__name__}: {e}")


def _error(message: str) -> ToolResult:
    return ToolResult(f"Error: {message}", is_error=True)
