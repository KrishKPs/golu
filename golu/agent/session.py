"""Save and resume conversations: .golu/sessions/<id>.json in the project directory.

A saved session records the system prompt and tool names it ran with. The
model's thinking blocks are only valid with exactly that prompt and tool set,
so if either changed (say, an MCP server was added), resuming drops the old
thinking blocks and keeps the visible conversation.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from golu.agent.context import Conversation

SESSIONS_DIR = Path(".golu") / "sessions"


@dataclass
class Session:
    id: str
    system: str
    tool_names: list[str]
    conversation: Conversation
    created: str = ""

    @classmethod
    def new(cls, system: str, tool_names: list[str]) -> "Session":
        return cls(
            id=datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            system=system,
            tool_names=tool_names,
            conversation=Conversation(),
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def adapt_to(self, system: str, tool_names: list[str]) -> bool:
        """Make a resumed session valid for the current setup. Returns True if it changed."""
        if system == self.system and tool_names == self.tool_names:
            return False
        for message in self.conversation.messages:
            message.raw = None  # rebuild from plain blocks, without old thinking
        self.system, self.tool_names = system, tool_names
        return True


def save(root: Path, session: Session) -> Path:
    path = root / SESSIONS_DIR / f"{session.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "id": session.id,
        "created": session.created,
        "system": session.system,
        "tool_names": session.tool_names,
        **session.conversation.to_dict(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash mid-save never leaves a half-written file
    return path


def load(root: Path, session_id: str) -> Session:
    path = root / SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved session '{session_id}' in {path.parent}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Session(
        id=data["id"],
        system=data["system"],
        tool_names=data["tool_names"],
        conversation=Conversation.from_dict(data),
        created=data.get("created", ""),
    )


def latest_id(root: Path) -> str | None:
    files = sorted((root / SESSIONS_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1].stem if files else None
