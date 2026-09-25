"""
Unit tests for the pieces of the pipeline that don't require network access
or a running Qdrant instance: chunking logic, evaluation metrics, and
deterministic point-id generation.

Run with: pytest
"""
from src.config import Settings
from src.data_prep import build_eval_queries, chunk_text, record_from_row
from src.embeddings import resolve_dense_vector_size
from src.qdrant_setup import DENSE_VECTOR_SIZE
from src.evaluate import precision_at_k, reciprocal_rank
from src.ingest import stable_point_id


def _settings(**overrides) -> Settings:
    base = dict(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="eu_reg_directives",
        embedding_provider="fastembed",
        cohere_api_key=None,
        dense_model_name="BAAI/bge-small-en-v1.5",
        sparse_model_name="prithivida/Splade_PP_en_v1",
        hf_dataset_name="coastalcph/lex_glue",
        hf_dataset_config="eurlex",
        hf_dataset_revision="main",
    )
    base.update(overrides)
    return Settings(**base)


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


def test_record_from_lex_glue_row_uses_eurovoc_title_and_stable_id():
    row = {
        "text": "Article 1\nThis Regulation lays down conservation measures.",
        "labels": [0, 2],
    }
    record = record_from_row(row, 4, ["fisheries", "transport", "environment"])
    assert record is not None
    assert record["doc_id"] == "eurlex-4"
    assert record["title"] == "fisheries, environment"
    assert record["text"].startswith("Article 1")


def test_record_from_row_keeps_legacy_celex_fields():
    row = {"celex_id": "32016R0679", "title": "GDPR", "celex_text": "Article 6 lawful basis."}
    record = record_from_row(row, 0, [])
    assert record["doc_id"] == "32016R0679"
    assert record["title"] == "GDPR"
    assert "lawful basis" in record["text"]


def test_record_from_row_skips_empty_text():
    assert record_from_row({"text": "  ", "labels": []}, 0, []) is None


def test_build_eval_queries_points_at_real_chunk_ids():
    chunks = [
        {
            "id": "eurlex-0::chunk0",
            "doc_id": "eurlex-0",
            "title": "environment",
            "text": (
                "Article 3\nMember States shall adopt penalties for infringement of this "
                "Regulation on environmental protection, waste shipments, and reporting duties "
                "of the competent authority. The Commission shall review application."
            ),
        },
        {
            "id": "eurlex-1::chunk0",
            "doc_id": "eurlex-1",
            "title": "fisheries",
            "text": (
                "The Commission shall monitor fishing quotas and conservation measures for "
                "marine biological resources. Member States shall report catch data each year "
                "to the authority designated under this Regulation before the season closes."
            ),
        },
    ]
    queries = build_eval_queries(chunks, limit=6)
    known_ids = {chunk["id"] for chunk in chunks}
    assert queries
    assert all(q["query"].strip() for q in queries)
    assert all(set(q["relevant_chunk_ids"]) <= known_ids for q in queries)
    assert any("penalt" in q["query"].lower() for q in queries)


def test_build_eval_queries_empty_chunks():
    assert build_eval_queries([], limit=5) == []


def test_bge_small_dense_size_is_384():
    assert DENSE_VECTOR_SIZE == 384
    assert resolve_dense_vector_size(_settings()) == DENSE_VECTOR_SIZE


def test_cohere_dense_size_is_1024():
    assert resolve_dense_vector_size(_settings(embedding_provider="cohere")) == 1024


def test_unknown_dense_model_is_rejected():
    try:
        resolve_dense_vector_size(_settings(dense_model_name="not-a-real-model"))
    except ValueError as exc:
        assert "not-a-real-model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_stable_point_id_is_deterministic():
    id1 = stable_point_id("CELEX:32016R0679::chunk4")
    id2 = stable_point_id("CELEX:32016R0679::chunk4")
    assert id1 == id2

    id3 = stable_point_id("CELEX:32016R0679::chunk5")
    assert id1 != id3
