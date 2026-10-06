"""Where the RAG server keeps its data.

Everything (the vector index and the downloaded embedding model) lives under
one data directory: $GOLU_DATA_DIR, or `.golu/` next to this repository.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return Path(os.environ.get("GOLU_DATA_DIR", REPO_ROOT / ".golu")).expanduser()


def db_path() -> Path:
    return data_dir() / "rag_db"


def models_dir() -> Path:
    return data_dir() / "models"
