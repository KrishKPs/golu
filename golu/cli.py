"""Entry point: `golu "task"` runs one task; `golu` alone starts an interactive session."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from golu.agent import session as sessions
from golu.agent.loop import Agent
from golu.agent.prompts import build_system_prompt, doc_tool_names
from golu.llm.base import LLMError, LLMProvider
from golu.mcp.config import ConfigError, load_servers
from golu.permissions import Permissions
from golu.tools import builtin_tools
from golu.tools.base import Tool
from golu.tools.paths import Workspace
from golu.ui import TerminalUI

app = typer.Typer(add_completion=False, help="Golu: an AI coding assistant with cited docs.")
console = Console()

# Tests replace this to inject a FakeProvider; it receives (model, effort, protected_tools).
ProviderFactory = Callable[[str, str, tuple[str, ...]], LLMProvider]
provider_factory: ProviderFactory | None = None


def _make_provider(model: str, effort: str, protected: tuple[str, ...]) -> LLMProvider:
    if provider_factory is not None:
        return provider_factory(model, effort, protected)
    from golu.llm.anthropic import AnthropicProvider, ContextEditing

    return AnthropicProvider(model, effort, context_editing=ContextEditing(exclude_tools=protected))


@app.command()
def main(
    task: Annotated[str | None, typer.Argument(help="Task to run. Omit for interactive.")] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Auto-approve edits and commands (denylist still applies)."
        ),
    ] = False,
    resume: Annotated[str | None, typer.Option(help="Resume a saved session by id.")] = None,
    cont: Annotated[
        bool, typer.Option("--continue", "-c", help="Resume the most recent session.")
    ] = False,
    model: Annotated[str, typer.Option(help="Model id.")] = "claude-opus-5-5",
    effort: Annotated[str, typer.Option(help="low, medium, high, xhigh, or max.")] = "high",
    no_mcp: Annotated[bool, typer.Option("--no-mcp", help="Don't start MCP servers.")] = False,
) -> None:
    root = Path.cwd().resolve()
    ui = TerminalUI(console)
    tools: list[Tool] = builtin_tools(Workspace(root))

    manager = None
    if not no_mcp:
        try:
            configs = load_servers(root)
        except ConfigError as e:
            console.print(f"[red]Config error:[/red] {e}")
            raise typer.Exit(code=2) from e
        if configs:
            from golu.mcp.client import MCPManager

            with console.status("Starting MCP servers…"):
                manager = MCPManager(log_dir=root / ".golu" / "logs")
                manager.connect_all(configs, root)
            for warning in manager.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning}")
            tools += manager.tools

    try:
        system = build_system_prompt(root, tools)
        session = _open_session(root, resume, cont, system, [t.name for t in tools])
        agent = Agent(
            _make_provider(model, effort, doc_tool_names(tools)),
            tools,
            Permissions(mode="auto" if yes else "ask", prompter=ui.ask_permission),
            system,
            conversation=session.conversation,
            events=ui.events(),
        )
        if task is not None:
            if not _turn(agent, task, ui, root, session):
                raise typer.Exit(code=1)
            return
        _repl(agent, ui, root, session)
    finally:
        if manager is not None:
            manager.close()


def _open_session(
    root: Path, resume: str | None, cont: bool, system: str, tool_names: list[str]
) -> sessions.Session:
    session_id = resume or (sessions.latest_id(root) if cont else None)
    if cont and session_id is None:
        console.print("[yellow]No saved session to continue; starting a new one.[/yellow]")
    if session_id is None:
        return sessions.Session.new(system, tool_names)
    try:
        session = sessions.load(root, session_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    if session.adapt_to(system, tool_names):
        console.print(
            "[dim]Tools changed since this session was saved; continuing without "
            "the model's earlier private reasoning.[/dim]"
        )
    console.print(
        f"[dim]Resumed session {session.id} ({len(session.conversation.messages)} messages).[/dim]"
    )
    return session


def _turn(agent: Agent, text: str, ui: TerminalUI, root: Path, session: sessions.Session) -> bool:
    """Run one user message. Returns False if the model call failed."""
    ui.begin_turn()
    try:
        result = agent.run(text)
    except LLMError as e:
        ui.finish()
        console.print(f"[red]Error:[/red] {e}")
        return False
    except KeyboardInterrupt:
        ui.finish()
        console.print("[yellow]Interrupted.[/yellow]")
        return True
    finally:
        sessions.save(root, session)
    ui.show_turn(result)
    return True


def _repl(agent: Agent, ui: TerminalUI, root: Path, session: sessions.Session) -> None:
    console.print(f"[bold]Golu[/bold] in {root}  [dim](session {session.id}; 'exit' to quit)[/dim]")
    while True:
        try:
            line = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if line in {"exit", "quit", "/exit", "/quit"}:
            break
        if line:
            _turn(agent, line, ui, root, session)
    console.print(f"[dim]Resume later with: golu --resume {session.id}[/dim]")
