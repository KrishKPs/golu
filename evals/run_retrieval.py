"""Retrieval eval: does search_docs return the right section for real questions?

    uv run python evals/run_retrieval.py                  # real local embedder
    uv run python evals/run_retrieval.py --save evals/results/after.json \
        --baseline evals/results/before.json            # compare with an earlier run

It indexes evals/corpus into a temporary in-memory store, then runs each query in
evals/retrieval/queries.jsonl through the same Retriever that search_docs uses.

Metrics (computed on what Golu actually receives, i.e. after the relevance cutoff):
- hit@1 / hit@3 / hit@5: an expected section is among the top 1 / 3 / 5 results.
- MRR: average of 1/rank of the first correct result (0 if missing).
- abstain: for questions the docs don't cover, the share that correctly return nothing.
- false empty: covered questions that wrongly returned nothing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_server.config import models_dir  # noqa: E402
from rag_server.embeddings import Embedder, HashEmbedder, LocalEmbedder  # noqa: E402
from rag_server.ingest import ingest  # noqa: E402
from rag_server.retrieve import DEFAULT_KEYWORD_WEIGHT, DEFAULT_MIN_SCORE, Retriever  # noqa: E402
from rag_server.store import DocStore, StoredChunk  # noqa: E402

CORPUS = ROOT / "evals" / "corpus"
QUERIES = ROOT / "evals" / "retrieval" / "queries.jsonl"


def matches(hit: StoredChunk, expected: list[list[str]]) -> bool:
    # Sources are relative to the corpus root, e.g. "quillstore/api-reference.md".
    return any(
        hit.source.endswith(source) and hit.section.rsplit(" > ", 1)[-1] == section
        for source, section in expected
    )


def evaluate(retriever: Retriever, queries: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    rows = []
    for q in queries:
        hits = retriever.search(q["query"], k=k)
        rank = next((i + 1 for i, h in enumerate(hits) if matches(h, q["expected"])), None)
        rows.append(
            {
                "query": q["query"],
                "covered": bool(q["expected"]),
                "rank": rank,
                "returned": len(hits),
                "top": [f"{h.source} § {h.section} ({h.score:.2f})" for h in hits[:3]],
            }
        )

    covered = [r for r in rows if r["covered"]]
    uncovered = [r for r in rows if not r["covered"]]

    def share(items: list[Any], pred: Any) -> float:
        return round(sum(1 for x in items if pred(x)) / len(items), 3) if items else 0.0

    return {
        "summary": {
            "queries": len(rows),
            "hit@1": share(covered, lambda r: r["rank"] == 1),
            "hit@3": share(covered, lambda r: r["rank"] is not None and r["rank"] <= 3),
            "hit@5": share(covered, lambda r: r["rank"] is not None and r["rank"] <= 5),
            "mrr": round(sum(1 / r["rank"] for r in covered if r["rank"]) / len(covered), 3),
            "abstain": share(uncovered, lambda r: r["returned"] == 0),
            "false_empty": share(covered, lambda r: r["returned"] == 0),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", choices=["local", "hash"], default="local")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--save", type=Path, help="Write results JSON here")
    parser.add_argument("--baseline", type=Path, help="Earlier results JSON to compare with")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every query")
    args = parser.parse_args()

    embedder: Embedder = LocalEmbedder(models_dir()) if args.embedder == "local" else HashEmbedder()
    store = DocStore(embedder)
    ingest(CORPUS, store)
    queries = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    retriever = Retriever(store, min_score=args.min_score, keyword_weight=args.keyword_weight)
    result = evaluate(retriever, queries, args.k)
    result["config"] = {
        "embedder": embedder.name,
        "min_score": args.min_score,
        "keyword_weight": args.keyword_weight,
        "k": args.k,
        "chunks": store.count(),
    }

    baseline = json.loads(args.baseline.read_text())["summary"] if args.baseline else None
    print(f"\nRetrieval eval: {result['config']}")
    for name, value in result["summary"].items():
        line = f"  {name:<12} {value}"
        if baseline and name in baseline and name != "queries":
            delta = value - baseline[name]
            line += f"   (baseline {baseline[name]}, {'+' if delta >= 0 else ''}{delta:.3f})"
        print(line)

    misses = [
        r
        for r in result["rows"]
        if (r["covered"] and r["rank"] != 1) or (not r["covered"] and r["returned"])
    ]
    shown = result["rows"] if args.verbose else misses
    if shown:
        print("\nNot ranked first / should have been empty:" if not args.verbose else "\nAll:")
        for r in shown:
            status = f"rank {r['rank']}" if r["covered"] else f"{r['returned']} results"
            print(f"  [{status}] {r['query']}")
            for t in r["top"]:
                print(f"        {t}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
