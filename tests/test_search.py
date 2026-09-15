"""
Unit tests for the pieces of the pipeline that don't require network access
or a running Qdrant instance: chunking logic, evaluation metrics, and
deterministic point-id generation.

Run with: pytest
"""
from src.data_prep import chunk_text
from src.evaluate import precision_at_k, reciprocal_rank
from src.ingest import stable_point_id


def test_chunk_text_splits_on_article_boundaries():
    text = (
        "Preamble text here.\n"
        "Article 1\nGeneral provisions establishing scope.\n"
        "Article 2\nDefinitions used throughout this Regulation.\n"
    )
    chunks = chunk_text(text, chunk_size=1000, overlap=50)
    assert any(c.startswith("Article 1") for c in chunks)
    assert any(c.startswith("Article 2") for c in chunks)


def test_chunk_text_respects_max_size_with_overlap():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_precision_at_k_all_relevant():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_partial_match():
    retrieved = ["a", "x", "y", "b"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_stable_point_id_is_deterministic():
    id1 = stable_point_id("CELEX:32016R0679::chunk4")
    id2 = stable_point_id("CELEX:32016R0679::chunk4")
    assert id1 == id2

    id3 = stable_point_id("CELEX:32016R0679::chunk5")
    assert id1 != id3
