"""run_command: runs a shell command in the project directory, with a timeout.

The permission layer checks the command (denylist + user approval) before
`run` is ever called. Credentials are removed from the child's environment,
and any secret that still shows up in the output is redacted.
"""

import os
import signal
import subprocess
from typing import Any

from golu.secrets import redact, scrubbed_env
from golu.tools.base import PermissionRequest, Tool, ToolError, ToolResult
from golu.tools.paths import Workspace

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600


class RunCommand(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the project directory and return its exit code and output. "
        "Use it for tests, builds, git status/diff, and similar. Commands time out "
        f"(default {DEFAULT_TIMEOUT}s, max {MAX_TIMEOUT}s) and must not need interactive input. "
        "Destructive commands (rm -rf, sudo, force-push, disk/network config) are blocked."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout_seconds": {"type": "integer", "description": "Timeout in seconds."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def permission_request(self, args: dict[str, Any]) -> PermissionRequest:
        command = args["command"].strip()
        if not command:
            raise ToolError("Command is empty.")
        return PermissionRequest("command", self.name, "Run command", command, command=command)

    def run(self, args: dict[str, Any]) -> ToolResult:
        timeout = min(max(1, args.get("timeout_seconds", DEFAULT_TIMEOUT)), MAX_TIMEOUT)
        # start_new_session puts the command in its own process group, so on
        # timeout we can kill it and everything it started.
        proc = subprocess.Popen(
            args["command"],
            shell=True,
            cwd=self.ws.root,
            env=scrubbed_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            return ToolResult(
                redact(_format(None, stdout, stderr) + f"\n[timed out after {timeout}s]"),
                is_error=True,
            )
        return ToolResult(redact(_format(proc.returncode, stdout, stderr)), proc.returncode != 0)


def _format(code: int | None, stdout: str, stderr: str) -> str:
    parts = [f"exit code: {code if code is not None else 'killed'}"]
    if stdout.strip():
        parts.append(f"--- stdout ---\n{stdout.rstrip()}")
    if stderr.strip():
        parts.append(f"--- stderr ---\n{stderr.rstrip()}")
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n".join(parts)
