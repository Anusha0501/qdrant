"""
Embedding generation for the hybrid pipeline.

- Dense vectors: FastEmbed (default, local ONNX inference) or Cohere API.
- Sparse vectors: Qdrant FastEmbed's SPLADE++ model (learned sparse, context
  aware — unlike raw BM25 term frequency). This FastEmbed release does not
  ship BGE-M3 sparse weights; SPLADE is the learned-expansion model, and
  `Qdrant/bm42-all-minilm-l6-v2-attentions` is the lighter alternative.

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


# Cohere embed-multilingual-v3.0. FastEmbed models report their own size.
COHERE_DENSE_SIZE = 1024


def resolve_dense_vector_size(settings: Settings | None = None) -> int:
    """
    Dimension written into the Qdrant dense named vector.

    BAAI/bge-small-en-v1.5 is 384. Cohere embed-multilingual-v3.0 is 1024.
    A hardcoded 384 would reject every Cohere vector at ingest time.
    """
    settings = settings or load_settings()
    if settings.embedding_provider == "cohere":
        return COHERE_DENSE_SIZE

    for model in TextEmbedding.list_supported_models():
        if model.get("model") == settings.dense_model_name:
            return int(model["dim"])
    known = ", ".join(sorted(m["model"] for m in TextEmbedding.list_supported_models())[:8])
    raise ValueError(
        f"DENSE_MODEL_NAME={settings.dense_model_name!r} is not a FastEmbed dense model. "
        f"Examples: {known}. Or set EMBEDDING_PROVIDER=cohere."
    )


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
