"""Task eval: run Golu end to end on fixed tasks and check the results.

    uv run python evals/run_tasks.py                 # all tasks in evals/tasks
    uv run python evals/run_tasks.py ttl_cache       # one task
    uv run python evals/run_tasks.py --save evals/results/tasks.json

Needs a real model (ANTHROPIC_API_KEY) and costs real money: roughly a few cents
to a dollar per run. Each task runs in a fresh temp directory with auto-approve
on (the denylist still applies) and the docs server connected.

A task file (evals/tasks/*.toml) has a prompt, optional starting [files], and
[checks]:
  file / file_must_match / file_must_not_match   regexes on a file after the run
  answer_must_match / answer_must_not_match      regexes on Golu's final reply
  must_cite      sources that the reply must cite AND that were actually retrieved
  command        must exit 0 ({python} = this Python)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from golu.agent.loop import Agent, TurnResult  # noqa: E402
from golu.agent.prompts import build_system_prompt, doc_tool_names  # noqa: E402
from golu.llm.base import LLMProvider  # noqa: E402
from golu.permissions import Permissions  # noqa: E402
from golu.tools import builtin_tools  # noqa: E402
from golu.tools.base import Tool  # noqa: E402
from golu.tools.paths import Workspace  # noqa: E402

TASKS = ROOT / "evals" / "tasks"


@dataclass
class TaskOutcome:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    tool_calls: int = 0
    seconds: float = 0.0
    output_tokens: int = 0
    answer: str = ""


def load_task(path: Path) -> dict[str, Any]:
    task = tomllib.loads(path.read_text())
    task.setdefault("name", path.stem)
    return task


def check(task: dict[str, Any], workdir: Path, result: TurnResult) -> list[str]:
    checks = task.get("checks", {})
    failures = []
    answer = result.text

    if "file" in checks:
        path = workdir / checks["file"]
        text = path.read_text() if path.exists() else ""
        if not path.exists():
            failures.append(f"{checks['file']} was not created")
        for pattern in checks.get("file_must_match", []):
            if not re.search(pattern, text):
                failures.append(f"{checks['file']} lacks /{pattern}/")
        for pattern in checks.get("file_must_not_match", []):
            if re.search(pattern, text):
                failures.append(f"{checks['file']} still has /{pattern}/")
    for pattern in checks.get("answer_must_match", []):
        if not re.search(pattern, answer):
            failures.append(f"answer lacks /{pattern}/")
    for pattern in checks.get("answer_must_not_match", []):
        if re.search(pattern, answer):
            failures.append(f"answer has /{pattern}/")
    retrieved = {c.source for c in result.citations}
    for source in checks.get("must_cite", []):
        if source not in answer:
            failures.append(f"answer does not cite {source}")
        if not any(source in r for r in retrieved):
            failures.append(f"{source} was never retrieved (citation not traceable)")
    if "command" in checks:
        cmd = checks["command"].format(python=sys.executable)
        proc = subprocess.run(
            cmd, shell=True, cwd=workdir, capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            failures.append(f"`{cmd}` exited {proc.returncode}: {proc.stdout[-300:]}")
    return failures


def run_task(
    task: dict[str, Any],
    make_provider: Callable[[tuple[str, ...]], LLMProvider],
    doc_tools: list[Tool],
) -> TaskOutcome:
    with tempfile.TemporaryDirectory(prefix=f"golu-eval-{task['name']}-") as tmp:
        workdir = Path(tmp).resolve()
        for name, content in task.get("files", {}).items():
            (workdir / name).parent.mkdir(parents=True, exist_ok=True)
            (workdir / name).write_text(content.lstrip("\n"))
        tools = builtin_tools(Workspace(workdir)) + doc_tools
        agent = Agent(
            make_provider(doc_tool_names(tools)),
            tools,
            Permissions(mode="auto"),
            build_system_prompt(workdir, tools),
        )
        calls = 0

        def count(_call: Any) -> None:
            nonlocal calls
            calls += 1

        agent.events.on_tool_start = count
        start = time.monotonic()
        try:
            result = agent.run(task["prompt"])
        except Exception as e:  # report and move on to the next task
            return TaskOutcome(task["name"], False, [f"crashed: {type(e).__name__}: {e}"])
        failures = check(task, workdir, result)
        return TaskOutcome(
            task["name"],
            not failures,
            failures,
            calls,
            round(time.monotonic() - start, 1),
            agent.conversation.total_output_tokens,
            result.text,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tasks", nargs="*", help="Task names (default: all)")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="high")
    args = parser.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Skipping task evals: set ANTHROPIC_API_KEY to run them (they call the real model).")
        return 0

    from mcp import StdioServerParameters

    from golu.llm.anthropic import AnthropicProvider, ContextEditing
    from golu.mcp.client import MCPManager
    from rag_server.config import data_dir

    paths = sorted(TASKS.glob("*.toml"))
    if args.tasks:
        paths = [p for p in paths if p.stem in args.tasks]
    _ensure_corpus_indexed()

    manager = MCPManager(log_dir=ROOT / ".golu" / "logs")
    manager.connect(
        "docs",
        StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag_server"],
            cwd=str(ROOT),
            env={"GOLU_DATA_DIR": str(data_dir())},
        ),
        "read_only",
    )
    if manager.warnings:
        print("\n".join(manager.warnings))
        return 1

    def make_provider(protected: tuple[str, ...]) -> LLMProvider:
        return AnthropicProvider(
            args.model, args.effort, context_editing=ContextEditing(exclude_tools=protected)
        )

    outcomes = []
    try:
        for path in paths:
            task = load_task(path)
            print(f"→ {task['name']} …", flush=True)
            outcome = run_task(task, make_provider, manager.tools)
            outcomes.append(outcome)
            mark = "PASS" if outcome.passed else "FAIL"
            print(f"  {mark}  {outcome.tool_calls} tool calls, {outcome.seconds}s")
            for f in outcome.failures:
                print(f"        - {f}")
    finally:
        manager.close()

    passed = sum(o.passed for o in outcomes)
    print(f"\n{passed}/{len(outcomes)} tasks passed")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(
                {
                    "config": {"model": args.model, "effort": args.effort},
                    "passed": passed,
                    "total": len(outcomes),
                    "tasks": [asdict(o) for o in outcomes],
                },
                indent=2,
            )
        )
    return 0 if passed == len(outcomes) else 1


def _ensure_corpus_indexed() -> None:
    """Make sure the eval docs are in the index the docs server reads."""
    from rag_server.config import db_path, models_dir
    from rag_server.embeddings import LocalEmbedder
    from rag_server.ingest import ingest
    from rag_server.store import DocStore

    store = DocStore(LocalEmbedder(models_dir()), db_path())
    ingest(ROOT / "evals" / "corpus" / "quillstore", store)


if __name__ == "__main__":
    raise SystemExit(main())
