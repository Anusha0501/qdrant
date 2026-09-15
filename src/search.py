"""
Hybrid search over the EU/UK regulatory collection.

Default mode ("hybrid") runs Dense + Learned-Sparse (SPLADE) candidates
through Qdrant's server-side Reciprocal Rank Fusion (RRF) in a single
`query_points` call using `prefetch`. Other modes exist purely so
evaluate.py can benchmark alternatives on identical data.

Usage:
    python -m src.search "obligations for providers of high-risk AI systems"
    python -m src.search "Article 22 GDPR automated decision-making" --mode dense
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Literal

from qdrant_client.http import models as qm

from src.config import DENSE_VECTOR_NAME, SPARSE_BM25_VECTOR_NAME, SPARSE_VECTOR_NAME, load_settings
from src.embeddings import get_bm25_embedder, get_embedders
from src.qdrant_setup import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SearchMode = Literal["dense", "dense_bm25", "hybrid"]


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


def _sparse_query_vector(indices: list[int], values: list[float]) -> qm.SparseVector:
    return qm.SparseVector(indices=indices, values=values)


def hybrid_search(
    query: str,
    mode: SearchMode = "hybrid",
    limit: int = 10,
    prefetch_limit: int = 50,
) -> list[SearchResult]:
    settings = load_settings()
    client = get_client()
    dense_embedder, sparse_embedder = get_embedders()

    dense_vec = dense_embedder.embed([query], is_query=True)[0]

    if mode == "dense":
        response = client.query_points(
            collection_name=settings.collection_name,
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
    else:
        if mode == "hybrid":
            sparse_vec = sparse_embedder.embed([query])[0]
            sparse_vector_name = SPARSE_VECTOR_NAME
        elif mode == "dense_bm25":
            sparse_vec = get_bm25_embedder().embed([query])[0]
            sparse_vector_name = SPARSE_BM25_VECTOR_NAME
        else:
            raise ValueError(f"Unknown search mode: {mode!r}")

        response = client.query_points(
            collection_name=settings.collection_name,
            prefetch=[
                qm.Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                qm.Prefetch(
                    query=_sparse_query_vector(sparse_vec.indices, sparse_vec.values),
                    using=sparse_vector_name,
                    limit=prefetch_limit,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

    return [
        SearchResult(
            chunk_id=p.payload["chunk_id"],
            doc_id=p.payload["doc_id"],
            title=p.payload.get("title", ""),
            text=p.payload["text"],
            score=p.score,
        )
        for p in response.points
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the EU regulatory hybrid search index.")
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", choices=["dense", "dense_bm25", "hybrid"], default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    results = hybrid_search(args.query, mode=args.mode, limit=args.limit)
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] score={r.score:.4f}  doc_id={r.doc_id}  title={r.title!r}")
        print(f"    {r.text[:280].strip()}...")


if __name__ == "__main__":
    main()
