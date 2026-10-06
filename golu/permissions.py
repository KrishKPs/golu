"""Decides whether a tool may edit a file or run a command.

Order of checks:
1. Commands go through the denylist first. A denied command is blocked even in
   auto-approve mode.
2. In "auto" mode everything else is allowed.
3. In "ask" mode the user is asked (showing the diff or exact command), unless
   they already chose "always" for that tool this session.

Read-only tools never reach this module; they have no permission request.
"""

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from golu.tools.base import PermissionRequest

Mode = Literal["ask", "auto"]
Answer = Literal["yes", "no", "always"]
Prompter = Callable[[PermissionRequest], Answer]


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


@dataclass
class Permissions:
    mode: Mode = "ask"
    prompter: Prompter | None = None
    always_allowed: set[str] = field(default_factory=set)  # tool names

    def decide(self, request: PermissionRequest) -> Decision:
        if request.command is not None:
            blocked = denylist_reason(request.command)
            if blocked:
                return Decision(False, f"blocked by the safety denylist: {blocked}")
        if self.mode == "auto" or request.tool in self.always_allowed:
            return Decision(True)
        if self.prompter is None:
            return Decision(
                False, "no one is available to approve it (run with --yes to auto-approve)"
            )
        answer = self.prompter(request)
        if answer == "always":
            self.always_allowed.add(request.tool)
        if answer == "no":
            return Decision(False, "the user declined")
        return Decision(True)


# --- Denylist ---------------------------------------------------------------

# Commands that are always blocked, whatever their arguments.
_BLOCKED_PROGRAMS = {
    "sudo": "sudo (runs as administrator)",
    "su": "su (switches user)",
    "doas": "doas (runs as administrator)",
    "mkfs": "formatting a disk",
    "fdisk": "disk partitioning",
    "sfdisk": "disk partitioning",
    "parted": "disk partitioning",
    "diskutil": "disk management",
    "wipefs": "wiping a disk",
    "shred": "destroying files",
    "ifconfig": "network configuration",
    "ip": "network configuration",
    "route": "network configuration",
    "networksetup": "network configuration",
    "iptables": "firewall configuration",
    "ip6tables": "firewall configuration",
    "pfctl": "firewall configuration",
    "ufw": "firewall configuration",
    "nmcli": "network configuration",
    "scutil": "network configuration",
    "shutdown": "shutting down the machine",
    "reboot": "rebooting the machine",
    "halt": "halting the machine",
}
# Prefixes that just run the next word as a command.
_WRAPPERS = {"env", "nohup", "time", "nice", "command", "exec", "xargs", "builtin", "caffeinate"}
_SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh"}
_SEPARATORS = re.compile(r"&&|\|\||[;|&\n()`]|\$\(")


def denylist_reason(command: str) -> str | None:
    """Why `command` is blocked, or None if it is allowed."""
    for segment in _SEPARATORS.split(command):
        reason = _check_segment(segment)
        if reason:
            return reason
    return None


def _check_segment(segment: str) -> str | None:
    try:
        words = shlex.split(segment, comments=True)
    except ValueError:
        words = segment.split()  # unbalanced quotes: fall back to a rough split
    # Skip leading VAR=value assignments and wrapper programs.
    while words and (re.fullmatch(r"\w+=.*", words[0]) or _program(words[0]) in _WRAPPERS):
        words = words[1:]
        # `env -i`, `nice -n 5` etc: drop their options too.
        while words and words[0].startswith("-"):
            words = words[1:]
    if not words:
        return None
    program, args = _program(words[0]), words[1:]

    if program in _BLOCKED_PROGRAMS or program.startswith("mkfs."):
        return _BLOCKED_PROGRAMS.get(program, "formatting a disk")
    if program in _SHELLS and "-c" in args:
        idx = args.index("-c")
        if idx + 1 < len(args):
            return denylist_reason(args[idx + 1])
    if program == "eval":
        return denylist_reason(" ".join(args))
    if program == "rm" and _is_recursive_force(args):
        return "rm -rf (recursive forced delete)"
    if program == "dd" and any(a.startswith("of=") for a in args):
        return "dd writing to a device or file"
    if program == "git" and _is_force_push(args):
        return "git push --force"
    return None


def _program(word: str) -> str:
    return word.rsplit("/", 1)[-1]  # /bin/rm -> rm


def _is_recursive_force(args: list[str]) -> bool:
    recursive = force = False
    for a in args:
        if a == "--":
            break
        if a in {"--recursive"}:
            recursive = True
        elif a in {"--force"}:
            force = True
        elif a.startswith("-") and not a.startswith("--"):
            recursive |= "r" in a or "R" in a
            force |= "f" in a
    return recursive and force


def _is_force_push(args: list[str]) -> bool:
    if "push" not in args:
        return False
    after = args[args.index("push") + 1 :]
    return any(
        a in {"-f", "--force", "--force-with-lease", "--force-if-includes", "--mirror"}
        or a.startswith("--force-with-lease=")
        or (a.startswith("-") and not a.startswith("--") and "f" in a)
        or a.startswith("+")
        for a in after
    )
