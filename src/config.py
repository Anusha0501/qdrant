"""
Centralized configuration for the EU Regulatory Compliance Search pipeline.

All configuration is sourced from environment variables (via .env in local
dev). Nothing sensitive is hardcoded — see .env.example for the full list of
supported variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op in prod if .env is absent; real secrets come from the environment


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name}={value!r} is invalid; expected one of {choices}")
    return value


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str

    embedding_provider: str
    cohere_api_key: str | None

    dense_model_name: str
    sparse_model_name: str

    hf_dataset_name: str
    hf_dataset_revision: str


def load_settings() -> Settings:
    embedding_provider = _require_choice(
        "EMBEDDING_PROVIDER",
        os.getenv("EMBEDDING_PROVIDER", "fastembed"),
        ("fastembed", "cohere"),
    )

    cohere_api_key = os.getenv("COHERE_API_KEY") or None
    if embedding_provider == "cohere" and not cohere_api_key:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=cohere requires COHERE_API_KEY to be set in the environment."
        )

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None

    return Settings(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=os.getenv("COLLECTION_NAME", "eu_reg_directives"),
        embedding_provider=embedding_provider,
        cohere_api_key=cohere_api_key,
        dense_model_name=os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        sparse_model_name=os.getenv("SPARSE_MODEL_NAME", "prithivida/Splade_PP_en_v1"),
        hf_dataset_name=os.getenv("HF_DATASET_NAME", "joelniklaus/eurlex"),
        hf_dataset_revision=os.getenv("HF_DATASET_REVISION", "main"),
    )


# Named vector identifiers used consistently across setup / ingest / search.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"        # learned sparse (SPLADE)
SPARSE_BM25_VECTOR_NAME = "sparse_bm25"  # classic BM25 sparse, kept only for the evaluation benchmark
