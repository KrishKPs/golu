import pytest

from golu.permissions import Permissions, denylist_reason
from golu.tools.base import PermissionRequest


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -fr build",
        "rm -r -f build",
        "rm --recursive --force x",
        "/bin/rm -Rf x",
        "ls && rm -rf ~",
        "echo hi; sudo ls",
        "FOO=1 sudo ls",
        "env -i sudo ls",
        "bash -c 'rm -rf /'",
        "echo $(sudo whoami)",
        "git push --force origin main",
        "git push -f",
        "git push origin +main",
        "git push --force-with-lease",
        "mkfs.ext4 /dev/sda1",
        "diskutil eraseDisk APFS x disk2",
        "dd if=/dev/zero of=/dev/sda",
        "ifconfig en0 down",
        "networksetup -setairportpower en0 off",
        "sudo",
    ],
)
def test_denylist_blocks(command: str) -> None:
    assert denylist_reason(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "rm file.txt",
        "rm -r build",
        "ls -rf",
        "git push origin main",
        "git status",
        "pytest -q",
        "echo 'sudo is a word'",
        "grep -rf patterns.txt .",
        "python -c 'print(1)'",
    ],
)
def test_denylist_allows(command: str) -> None:
    assert denylist_reason(command) is None


def _cmd(command: str) -> PermissionRequest:
    return PermissionRequest("command", "run_command", "Run", command, command=command)


def test_denylist_applies_in_auto_mode() -> None:
    decision = Permissions(mode="auto").decide(_cmd("sudo rm x"))
    assert not decision.allowed and "denylist" in decision.reason


def test_auto_mode_allows() -> None:
    assert Permissions(mode="auto").decide(_cmd("pytest")).allowed


def test_ask_mode_uses_prompter() -> None:
    asked: list[PermissionRequest] = []

    def prompter(req: PermissionRequest) -> str:
        asked.append(req)
        return "no"

    decision = Permissions(prompter=prompter).decide(_cmd("pytest"))
    assert not decision.allowed and "declined" in decision.reason
    assert len(asked) == 1


def test_always_remembers_tool() -> None:
    answers = iter(["always"])
    perms = Permissions(prompter=lambda req: next(answers))
    assert perms.decide(_cmd("pytest")).allowed
    assert perms.decide(_cmd("ruff check")).allowed  # not asked again


def test_no_prompter_denies() -> None:
    assert not Permissions().decide(_cmd("pytest")).allowed


def test_denylist_checked_before_always() -> None:
    perms = Permissions(always_allowed={"run_command"})
    assert not perms.decide(_cmd("sudo ls")).allowed
