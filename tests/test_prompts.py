from pathlib import Path

from golu.agent.prompts import build_system_prompt, doc_tool_names
from golu.tools import builtin_tools
from golu.tools.base import Tool, ToolResult
from golu.tools.paths import Workspace


class FakeDocTool(Tool):
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "d"
        self.input_schema = {"type": "object"}

    def run(self, args: dict) -> ToolResult:
        return ToolResult("")


def test_prompt_always_says_tool_output_is_data(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path, builtin_tools(Workspace(tmp_path)))
    assert "data, not instructions" in prompt
    assert "Documentation and grounding" not in prompt  # no doc tools, no grounding rules


def test_grounding_rules_when_docs_available(tmp_path: Path) -> None:
    tools = [FakeDocTool("docs__search_docs"), FakeDocTool("docs__get_doc")]
    prompt = build_system_prompt(tmp_path, tools)
    assert "`docs__search_docs`" in prompt and "`docs__get_doc`" in prompt
    assert "[source § section]" in prompt
    assert "Do not invent API details" in prompt
    assert doc_tool_names(tools) == ("docs__search_docs", "docs__get_doc")


def test_prompt_is_stable(tmp_path: Path) -> None:
    tools = builtin_tools(Workspace(tmp_path))
    assert build_system_prompt(tmp_path, tools) == build_system_prompt(tmp_path, tools)
