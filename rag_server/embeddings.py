"""Embedding models: turn text into vectors so similar meanings are close together.

`LocalEmbedder` runs all-MiniLM-L6-v2 locally through ONNX (no API calls).
The model (~80 MB) downloads once into the data directory and is checked
against a fixed SHA-256 hash. `HashEmbedder` is a tiny deterministic stand-in
for tests.
"""

import hashlib
import math
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    name = "all-MiniLM-L6-v2"

    def __init__(self, models_dir: Path) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        class _Model(ONNXMiniLM_L6_V2):
            DOWNLOAD_PATH = models_dir / "all-MiniLM-L6-v2"

        self._model = _Model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # The first call may download the model. Keep any progress output off
        # stdout: the MCP server uses stdout for protocol messages.
        with redirect_stdout(sys.stderr):
            vectors = self._model(texts)
        return [[float(x) for x in v] for v in vectors]


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size vector. Only for tests."""

    name = "hash-test"

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
