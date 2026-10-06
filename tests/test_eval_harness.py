"""The task-eval harness itself, driven by a fake model (no API calls)."""

from pathlib import Path

from evals.run_tasks import TASKS, check, load_task, run_task
from golu.agent.loop import TurnResult
from golu.llm.base import ToolCall
from golu.llm.fake import FakeProvider
from golu.tools.base import Citation


def test_fix_bug_task_passes_when_fixed() -> None:
    fix = ToolCall(
        "1",
        "edit_file",
        {"path": "mathutil.py", "old_string": "len(values) - 1", "new_string": "len(values)"},
    )
    provider = FakeProvider([[fix], "Fixed the off-by-one in mean()."])
    outcome = run_task(load_task(TASKS / "fix_bug.toml"), lambda _: provider, [])
    assert outcome.passed, outcome.failures
    assert outcome.tool_calls == 1


def test_fix_bug_task_fails_when_not_fixed() -> None:
    provider = FakeProvider(["I looked but changed nothing."])
    outcome = run_task(load_task(TASKS / "fix_bug.toml"), lambda _: provider, [])
    assert not outcome.passed
    assert any("exited" in f for f in outcome.failures)


def test_abstain_checks() -> None:
    task = load_task(TASKS / "abstain_replication.toml")
    honest = TurnResult("The indexed docs don't cover replication.", "end_turn")
    invented = TurnResult("Call store.replicate(host) to set it up.", "end_turn")
    assert check(task, Path("."), honest) == []
    assert check(task, Path("."), invented)


def test_citation_must_be_traceable(tmp_path: Path) -> None:
    task = {"name": "t", "prompt": "", "checks": {"must_cite": ["errors.md"]}}
    cited_not_retrieved = TurnResult("See [errors.md § ConflictError].", "end_turn")
    assert any("never retrieved" in f for f in check(task, tmp_path, cited_not_retrieved))
    ok = TurnResult(
        "See [errors.md § ConflictError].",
        "end_turn",
        [Citation("quillstore/errors.md", "Errors > ConflictError")],
    )
    assert check(task, tmp_path, ok) == []


def test_all_task_files_load() -> None:
    tasks = [load_task(p) for p in sorted(TASKS.glob("*.toml"))]
    assert len(tasks) == 5 and all(t["prompt"] and t["checks"] for t in tasks)


def test_abstain_accepts_common_phrasings() -> None:
    task = load_task(TASKS / "abstain_replication.toml")
    for answer in [
        "The docs don’t cover replication.",
        "Replication isn't documented in the indexed docs.",
        "I couldn't find anything about replication.",
        "The documentation does not mention replication.",
    ]:
        assert check(task, Path("."), TurnResult(answer, "end_turn")) == [], answer
