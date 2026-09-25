"""
Creates the Qdrant multi-vector collection used for hybrid search:
  - a named dense vector ("dense") with cosine distance
  - a named sparse vector ("sparse") for learned-sparse (SPLADE) retrieval

Usage:
    python -m src.qdrant_setup
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import (
    DENSE_VECTOR_NAME,
    SPARSE_BM25_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    load_settings,
)
from src.embeddings import resolve_dense_vector_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default dense size for BAAI/bge-small-en-v1.5. create_collection resolves
# the live size from the configured model (Cohere multilingual is 1024).
DENSE_VECTOR_SIZE = 384


def get_client() -> QdrantClient:
    settings = load_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def create_collection(
    client: QdrantClient,
    collection_name: str,
    recreate: bool = False,
    include_bm25: bool = True,
    dense_size: int | None = None,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and not recreate:
        logger.info("Collection %s already exists — skipping creation.", collection_name)
        return
    if exists and recreate:
        logger.warning("Recreating collection %s (all existing data will be dropped).", collection_name)
        client.delete_collection(collection_name)

    vector_size = dense_size if dense_size is not None else resolve_dense_vector_size()

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False),
            ),
            **(
                {
                    # Kept only so evaluate.py can benchmark Dense+BM25 against
                    # Dense+Learned-Sparse on identical data. Not used by search.py's
                    # default hybrid mode.
                    SPARSE_BM25_VECTOR_NAME: qm.SparseVectorParams(
                        index=qm.SparseIndexParams(on_disk=False),
                    )
                }
                if include_bm25
                else {}
            ),
        },
    )

    # Payload indexes speed up filtered hybrid queries (e.g. by doc_id / title).
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )

    logger.info(
        "Created collection %s with dense size %d plus sparse named vectors.",
        collection_name,
        vector_size,
    )


def main() -> None:
    settings = load_settings()
    client = get_client()
    create_collection(client, settings.collection_name)


if __name__ == "__main__":
    main()
