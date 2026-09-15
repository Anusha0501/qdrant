"""
Evaluation harness comparing:
  - Dense-only
  - Dense + BM25 (classic lexical) hybrid, RRF-fused
  - Dense + Learned Sparse (SPLADE) hybrid, RRF-fused  [this project's default]

on Precision@K and Mean Reciprocal Rank (MRR), against a labeled query set.

Expected input format (--queries), one JSON object per line:
    {"query": "...", "relevant_chunk_ids": ["CELEX:32016R0679::chunk4", ...]}

Usage:
    python -m src.evaluate --queries data/eval_queries.jsonl --k 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.search import SearchMode, hybrid_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES: list[SearchMode] = ["dense", "dense_bm25", "hybrid"]
MODE_LABELS = {
    "dense": "Dense-only",
    "dense_bm25": "Dense + BM25",
    "hybrid": "Dense + Learned Sparse (SPLADE)",
}


def load_eval_queries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(queries: list[dict], k: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}

    for mode in MODES:
        precisions, rrs = [], []
        for item in queries:
            relevant = set(item["relevant_chunk_ids"])
            retrieved = [r.chunk_id for r in hybrid_search(item["query"], mode=mode, limit=max(k, 10))]
            precisions.append(precision_at_k(retrieved, relevant, k))
            rrs.append(reciprocal_rank(retrieved, relevant))

        results[mode] = {
            f"precision@{k}": sum(precisions) / len(precisions) if precisions else 0.0,
            "mrr": sum(rrs) / len(rrs) if rrs else 0.0,
        }

    return results


def print_report(results: dict[str, dict[str, float]], k: int) -> None:
    header = f"{'Model':38s} {'Precision@' + str(k):>14s} {'MRR':>10s}"
    print(header)
    print("-" * len(header))
    for mode in MODES:
        metrics = results[mode]
        print(f"{MODE_LABELS[mode]:38s} {metrics[f'precision@{k}']:>14.3f} {metrics['mrr']:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hybrid search configurations.")
    parser.add_argument("--queries", type=Path, default=Path("data/eval_queries.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    queries = load_eval_queries(args.queries)
    if not queries:
        logger.warning("No evaluation queries found in %s.", args.queries)
        return

    results = evaluate(queries, args.k)
    print_report(results, args.k)


if __name__ == "__main__":
    main()
