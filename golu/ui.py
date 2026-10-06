"""Terminal output: streamed replies, tool activity, permission prompts, citations."""

import json
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax

from golu.agent.loop import AgentEvents, TurnResult
from golu.diffs import diff_stats
from golu.llm.base import ToolCall
from golu.permissions import Answer
from golu.tools.base import Citation, PermissionRequest, ToolResult

PERMISSION_PROMPT = "[yellow]Allow?[/] [bold]y[/]es / [bold]n[/]o / [bold]a[/]lways for this tool: "


class TerminalUI:
    def __init__(self, console: Console) -> None:
        self.console = console
        self._status: Status | None = None
        self._live: Live | None = None
        self._buffer = ""
        self.streamed = False  # did any reply text stream this turn?

    def begin_turn(self) -> None:
        self.streamed = False

    def events(self) -> AgentEvents:
        return AgentEvents(
            on_text=self.on_text,
            on_tool_start=self.on_tool_start,
            on_tool_end=self.on_tool_end,
            on_model_start=self.on_model_start,
        )

    # --- streaming ---

    def on_model_start(self) -> None:
        self._end_text()
        self._status = self.console.status("Thinking…", spinner="dots")
        self._status.start()

    def on_text(self, text: str) -> None:
        self.streamed = True
        self._stop_status()
        if self._live is None:
            self._buffer = ""
            self._live = Live(
                console=self.console, refresh_per_second=12, vertical_overflow="visible"
            )
            self._live.start()
        self._buffer += text
        self._live.update(Markdown(self._buffer))

    def finish(self) -> None:
        """Call after each turn and on errors."""
        self._stop_status()
        self._end_text()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _end_text(self) -> None:
        if self._live is not None:
            self._live.stop()  # leaves the final rendering on screen
            self._live = None
            if not self.console.is_terminal:
                self.console.line()  # piped output: Live doesn't end the line itself

    # --- tools ---

    def on_tool_start(self, call: ToolCall) -> None:
        self.finish()
        self.console.print(
            f"[bold magenta]●[/] [bold]{call.name}[/] [dim]{_summarize(call.arguments)}[/]"
        )

    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        if result.citations and not result.is_error:
            first = result.citations[0]
            self.console.print(
                f"  [dim]⎿ {len(result.citations)} doc section(s), top: "
                f"{_escape(first.source)} § {_escape(first.section)}[/]"
            )
            return
        lines = result.content.strip().splitlines() or [""]
        first = lines[0][:120]
        more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        style = "red" if result.is_error else "dim"
        self.console.print(f"  [{style}]⎿ {_escape(first)}{more}[/]")

    def ask_permission(self, request: PermissionRequest) -> Answer:
        self.finish()
        if request.kind == "edit":
            added, removed = diff_stats(request.detail)
            title = f"{request.summary}  [green]+{added}[/] [red]-{removed}[/]"
            body: Any = Syntax(
                request.detail or "(no changes)", "diff", theme="ansi_dark", word_wrap=True
            )
        elif request.kind == "command":
            title, body = (
                request.summary,
                Syntax(request.detail, "bash", theme="ansi_dark", word_wrap=True),
            )
        else:
            title, body = request.summary, Syntax(request.detail, "json", theme="ansi_dark")
        self.console.print(Panel(body, title=title, title_align="left", border_style="yellow"))
        while True:
            try:
                reply = self.console.input(PERMISSION_PROMPT).strip().lower()
            except EOFError:
                return "no"
            if reply in {"y", "yes"}:
                return "yes"
            if reply in {"n", "no", ""}:
                return "no"
            if reply in {"a", "always"}:
                return "always"

    # --- results ---

    def show_turn(self, result: TurnResult) -> None:
        self.finish()
        if not self.streamed and result.text:
            self.console.print(Markdown(result.text))
        if result.stop_reason == "refusal":
            self.console.print("[yellow]The model declined this request.[/]")
        if result.citations:
            self.console.print(_sources_panel(result.citations, result.text))


def _sources_panel(citations: list[Citation], answer: str) -> Panel:
    """Every doc chunk retrieved this turn. ✓ marks the ones the answer cites."""
    lines = []
    for c in citations:
        cited = c.source in answer and c.section.rsplit(" > ", 1)[-1] in answer
        mark = "[green]✓[/]" if cited else "[dim]·[/]"
        lines.append(f"{mark} {_escape(c.label())}")
    return Panel("\n".join(lines), title="Docs retrieved", title_align="left", border_style="blue")


def _summarize(args: dict[str, Any]) -> str:
    for key in ("path", "command", "query", "pattern", "source"):
        if key in args:
            return _escape(str(args[key])[:100])
    return _escape(json.dumps(args)[:100])


def _escape(text: str) -> str:
    return text.replace("[", r"\[")
