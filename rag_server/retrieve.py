"""Query -> the most relevant chunks, with source metadata for citations.

Two steps:
1. Relevance cutoff on the embedding (meaning) score: chunks below `min_score`
   are dropped, so an unrelated question returns nothing instead of the "least
   bad" chunks. That is what lets Golu say "the docs don't cover this".
2. Ranking: the survivors are ordered by meaning score plus a small bonus for
   sharing words with the question. Embeddings sometimes miss exact terms
   ("zstd", "python objects") that keyword overlap catches.

Both settings were tuned on evals/retrieval; re-run it after changing them.
"""

import re

from rag_server.store import DocStore, StoredChunk

DEFAULT_MIN_SCORE = 0.20
DEFAULT_KEYWORD_WEIGHT = 0.15
CANDIDATES = 20  # how many embedding matches to consider before re-ranking

_STOPWORDS = set(
    "a an and are as at be by can do does for from get how i if in is it its my of on or "
    "so that the this to use what when where which with without you your there".split()
)


def _terms(text: str) -> set[str]:
    """Lowercase words minus stopwords, with a crude plural/tense strip."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 1}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def keyword_overlap(query: str, chunk: StoredChunk) -> float:
    """Share of the question's words that appear in the chunk (0..1)."""
    q = _terms(query)
    if not q:
        return 0.0
    return len(q & _terms(f"{chunk.section} {chunk.text}")) / len(q)


class Retriever:
    def __init__(
        self,
        store: DocStore,
        min_score: float = DEFAULT_MIN_SCORE,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ) -> None:
        self.store = store
        self.min_score = min_score
        self.keyword_weight = keyword_weight

    def search(self, query: str, k: int = 5, library: str | None = None) -> list[StoredChunk]:
        k = max(1, min(k, 20))
        candidates = self.store.query(query, max(k, CANDIDATES), library)
        relevant = [c for c in candidates if c.score is not None and c.score >= self.min_score]
        if self.keyword_weight:
            relevant.sort(
                key=lambda c: (c.score or 0) + self.keyword_weight * keyword_overlap(query, c),
                reverse=True,
            )
        return relevant[:k]

    def get_doc(self, source: str, section: str | None = None) -> list[StoredChunk]:
        """A whole document, or one section of it (including its subsections)."""
        chunks = self.store.by_source(source)
        if section is None:
            return chunks
        wanted = section.strip().lower()
        return [
            c
            for c in chunks
            if c.section.lower() == wanted
            or c.section.lower().startswith(wanted + " > ")
            or c.section.lower().rsplit(" > ", 1)[-1] == wanted
        ]
