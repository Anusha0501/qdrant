"""
Embedding generation for the hybrid pipeline.

- Dense vectors: FastEmbed (default, local ONNX inference) or Cohere API.
- Sparse vectors: Qdrant FastEmbed's SPLADE++ model (learned sparse, context
  aware — unlike raw BM25 term frequency).

Both embedders are wrapped behind a small interface so `ingest.py` and
`search.py` don't need to know which provider is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_text_embedding import SparseEmbedding

from src.config import Settings, load_settings


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class DenseEmbedder:
    """Dense embedder abstraction over FastEmbed (local) or Cohere (API)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.embedding_provider == "fastembed":
            self._model = TextEmbedding(model_name=settings.dense_model_name)
            self._cohere_client = None
        else:
            import cohere  # imported lazily so fastembed-only installs still work

            self._model = None
            self._cohere_client = cohere.Client(api_key=settings.cohere_api_key)

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if self._model is not None:
            return [vec.tolist() for vec in self._model.embed(texts)]

        # Cohere distinguishes query vs. document embedding input types.
        input_type = "search_query" if is_query else "search_document"
        resp = self._cohere_client.embed(
            texts=texts,
            model="embed-multilingual-v3.0",
            input_type=input_type,
        )
        return resp.embeddings


class SparseEmbedder:
    """Generic wrapper around a FastEmbed sparse model (SPLADE or BM25)."""

    def __init__(self, model_name: str):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        results: list[SparseEmbedding] = list(self._model.embed(texts))
        return [
            SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
            for r in results
        ]


@lru_cache(maxsize=1)
def get_embedders() -> tuple[DenseEmbedder, SparseEmbedder]:
    """Dense embedder + learned-sparse (SPLADE) embedder — the pair used for ingest/search."""
    settings = load_settings()
    return DenseEmbedder(settings), SparseEmbedder(settings.sparse_model_name)


@lru_cache(maxsize=1)
def get_bm25_embedder() -> SparseEmbedder:
    """Classic BM25 sparse embedder, used only by the evaluation benchmark for comparison."""
    return SparseEmbedder("Qdrant/bm25")
