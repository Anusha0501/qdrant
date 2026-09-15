"""
Embeds chunked regulatory text (dense + sparse) and upserts into the Qdrant
collection created by qdrant_setup.py.

Usage:
    python -m src.ingest --chunks data/chunks.jsonl --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

from qdrant_client.http import models as qm
from tqdm import tqdm

from src.config import (
    DENSE_VECTOR_NAME,
    SPARSE_BM25_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    load_settings,
)
from src.embeddings import get_bm25_embedder, get_embedders
from src.qdrant_setup import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_point_id(chunk_id: str) -> str:
    """Deterministic UUID from the chunk's natural key, so re-runs upsert in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest(chunks: list[dict], batch_size: int = 64, include_bm25: bool = True) -> None:
    settings = load_settings()
    client = get_client()
    dense_embedder, sparse_embedder = get_embedders()
    bm25_embedder = get_bm25_embedder() if include_bm25 else None

    if not client.collection_exists(settings.collection_name):
        raise RuntimeError(
            f"Collection {settings.collection_name!r} does not exist. "
            "Run `python -m src.qdrant_setup` first."
        )

    for batch in tqdm(list(batched(chunks, batch_size)), desc="ingesting"):
        texts = [c["text"] for c in batch]
        dense_vecs = dense_embedder.embed(texts, is_query=False)
        sparse_vecs = sparse_embedder.embed(texts)
        bm25_vecs = bm25_embedder.embed(texts) if bm25_embedder else [None] * len(texts)

        points = []
        for chunk, dense_vec, sparse_vec, bm25_vec in zip(batch, dense_vecs, sparse_vecs, bm25_vecs):
            vector = {
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: qm.SparseVector(
                    indices=sparse_vec.indices, values=sparse_vec.values
                ),
            }
            if bm25_vec is not None:
                vector[SPARSE_BM25_VECTOR_NAME] = qm.SparseVector(
                    indices=bm25_vec.indices, values=bm25_vec.values
                )
            points.append(
                qm.PointStruct(
                    id=stable_point_id(chunk["id"]),
                    vector=vector,
                    payload={
                        "chunk_id": chunk["id"],
                        "doc_id": chunk["doc_id"],
                        "title": chunk.get("title", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "text": chunk["text"],
                    },
                )
            )

        client.upsert(collection_name=settings.collection_name, points=points)

    logger.info("Ingested %d chunks into %s.", len(chunks), settings.collection_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and upsert chunks into Qdrant.")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        logger.warning("No chunks found in %s — nothing to ingest.", args.chunks)
        return
    ingest(chunks, args.batch_size)


if __name__ == "__main__":
    main()
