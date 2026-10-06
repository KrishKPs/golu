"""Loads the MCP server list from golu.toml in the project directory.

    [[mcp_servers]]
    name = "docs"                     # tools appear as docs__search_docs, ...
    command = "uv"
    args = ["run", "python", "-m", "rag_server"]
    cwd = "/path/to/golu"             # optional, default: the project directory
    env = { GOLU_DATA_DIR = "..." }   # optional extra environment (no secrets!)
    trust = "read_only"               # "ask" (default), "read_only", or "all"

trust controls approval for this server's tools:
- "ask": every call asks the user.
- "read_only": tools the server marks read-only run without asking; others ask.
- "all": no approval needed. Only for servers you fully trust.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Trust = Literal["ask", "read_only", "all"]
CONFIG_FILE = "golu.toml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    trust: Trust = "ask"


def load_servers(project_root: Path) -> list[ServerConfig]:
    path = project_root / CONFIG_FILE
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e

    servers = []
    for i, raw in enumerate(data.get("mcp_servers", [])):
        where = f"{path} mcp_servers[{i}]"
        name, command = raw.get("name"), raw.get("command")
        if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"{where}: 'name' must be letters, digits, '-' or '_'.")
        if not isinstance(command, str):
            raise ConfigError(f"{where}: 'command' is required.")
        trust = raw.get("trust", "ask")
        if trust not in ("ask", "read_only", "all"):
            raise ConfigError(f"{where}: 'trust' must be ask, read_only, or all.")
        servers.append(
            ServerConfig(
                name=name,
                command=command,
                args=[str(a) for a in raw.get("args", [])],
                env={str(k): str(v) for k, v in raw.get("env", {}).items()},
                cwd=raw.get("cwd"),
                trust=trust,
            )
        )
    return servers
