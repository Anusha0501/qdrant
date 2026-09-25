"""
Dataset preparation for EU regulatory / legal directives.

Loads English EU legislation from Hugging Face. The default corpus is
LexGLUE's ``eurlex`` config (``coastalcph/lex_glue``). The older
``joelniklaus/eurlex`` repo was removed from the Hub and returns 404, which
blocked every later step.

Also writes ``data/eval_queries.jsonl``: a labeled query set whose gold chunk
ids are aligned to the chunks just produced, so Precision@K / MRR can run.

Usage:
    python -m src.data_prep --max-docs 500 --out data/chunks.jsonl
    python -m src.data_prep --from-chunks data/chunks.jsonl --eval-out data/eval_queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm

from src.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Roughly matches EU-style article/recital headers so we don't split mid-article
# where avoidable, e.g. "Article 6", "Recital 47", "Art. 22(1)".
ARTICLE_BOUNDARY_RE = re.compile(
    r"(?=\n\s*(Article\s+\d+|Art\.\s*\d+|Recital\s+\d+)\b)", re.IGNORECASE
)
ARTICLE_MENTION_RE = re.compile(
    r"\b(Article\s+\d+(?:\(\d+\))?|Recital\s+\d+)\b", re.IGNORECASE
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")

# Function words and statutory boilerplate. Domain terms stay in the query.
_STOPWORDS = frozenset(
    """
    the and shall this that with from regulation directive article recital
    member states state european union commission having regard whereas
    pursuant accordance thereof such under into their have been will may
    must also each other where which when upon after before these those
    within without including provided following between among paragraph
    point annex chapter section hereby thereof therein hereby pursuant
    thereof adopted concerning relating pursuant official journal
    through granted resources bring force laws necessary provisions
    administrative following publication within application those
    effective proportionate dissuasive which threatens distort
    favouring certain incompatible internal market form notifications
    setting twentieth measures biological
    """.split()
)

# Known-item retrieval is filled from the chunks themselves. These curated
# queries are added only when the loaded text actually contains the anchors,
# so a run against LexGLUE still gets legal-looking questions with real gold ids.
CURATED_QUERIES: list[dict] = [
    {
        "query": "What penalties must Member States lay down for infringement of this Regulation?",
        "all_of": ["penalt"],
        "any_of": ["member state", "infringe", "sanction"],
    },
    {
        "query": "When does this act enter into force, and from when does it apply?",
        "all_of": ["enter into force"],
        "any_of": ["apply", "publication", "official journal"],
    },
    {
        "query": "Which powers does the Commission have to adopt delegated or implementing acts?",
        "all_of": ["commission"],
        "any_of": ["delegated", "implementing act"],
    },
    {
        "query": "How must Member States transpose this Directive into national law?",
        "all_of": ["member state"],
        "any_of": ["transpose", "bring into force", "national law"],
    },
    {
        "query": "What reporting or monitoring duties are imposed on the Commission?",
        "all_of": ["commission"],
        "any_of": ["report", "monitor", "review"],
    },
    {
        "query": "What conservation or quota rules apply to fisheries and marine resources?",
        "all_of": ["fish"],
        "any_of": ["quota", "vessel", "conservation", "catch"],
    },
    {
        "query": "Which rules govern state aid and competition between undertakings?",
        "all_of": ["aid"],
        "any_of": ["competition", "undertaking", "commission", "compatible"],
    },
]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Chunk text on article/recital boundaries first, then apply a sliding
    character window within any oversized segment. Overlap preserves context
    across chunk boundaries (important for cross-referenced legal clauses).
    """
    if not text or not text.strip():
        return []

    segments = ARTICLE_BOUNDARY_RE.split(text) or [text]
    chunks: list[str] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) <= chunk_size:
            chunks.append(segment)
            continue

        start = 0
        while start < len(segment):
            end = min(start + chunk_size, len(segment))
            chunks.append(segment[start:end].strip())
            if end == len(segment):
                break
            start = end - overlap

    return [c for c in chunks if c]


def _label_names(dataset) -> list[str]:
    features = getattr(dataset, "features", None)
    if not features or "labels" not in features:
        return []
    feature = features["labels"]
    inner = getattr(feature, "feature", None)
    names = getattr(inner, "names", None) if inner is not None else None
    if not names:
        names = getattr(feature, "names", None)
    return list(names or [])


def record_from_row(row: dict, index: int, label_names: list[str]) -> dict | None:
    """
    Normalize one source row.

    LexGLUE eurlex rows have ``text`` and multi-label ``labels`` (EuroVoc
    class ids). Older EUR-Lex dumps used ``celex_id`` / ``title`` / ``celex_text``.
    Both shapes are accepted so a locally cached file still loads.
    """
    text = row.get("text") or row.get("celex_text") or ""
    if isinstance(text, list):
        text = "\n".join(str(part) for part in text)
    text = str(text)
    if not text.strip():
        return None

    raw_labels = row.get("labels") or []
    if isinstance(raw_labels, int):
        raw_labels = [raw_labels]
    names: list[str] = []
    for label in raw_labels:
        if isinstance(label, str) and label.strip():
            names.append(label.strip())
        elif isinstance(label, int) and 0 <= label < len(label_names):
            names.append(str(label_names[label]))

    title = str(row.get("title") or "")
    if not title.strip() and names:
        title = ", ".join(names)

    doc_id = row.get("celex_id") or row.get("id")
    if doc_id is None or str(doc_id).strip() == "":
        doc_id = f"eurlex-{index}"

    return {
        "doc_id": str(doc_id),
        "title": title[:300],
        "text": text,
    }


def load_source_dataset(dataset_name: str, config_name: str | None, revision: str):
    """Load the train split. ``config_name`` is required for LexGLUE (``eurlex``)."""
    kwargs = {"split": "train"}
    if revision:
        kwargs["revision"] = revision
    logger.info(
        "Loading dataset %s config=%s revision=%s",
        dataset_name,
        config_name or "(none)",
        revision or "(default)",
    )
    if config_name:
        return load_dataset(dataset_name, config_name, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def iter_eurlex_records(
    dataset_name: str,
    revision: str,
    max_docs: int | None,
    config_name: str | None = None,
) -> Iterator[dict]:
    dataset = load_source_dataset(dataset_name, config_name, revision)
    names = _label_names(dataset)
    kept = 0
    for index, row in enumerate(dataset):
        if max_docs is not None and kept >= max_docs:
            break
        record = record_from_row(row, index, names)
        if record is None:
            continue
        yield record
        kept += 1


def iter_local_files(directory: Path) -> Iterator[dict]:
    if not directory.exists():
        return
    for path in sorted(directory.glob("**/*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        yield {
            "doc_id": path.stem,
            "title": path.stem.replace("_", " "),
            "text": path.read_text(encoding="utf-8", errors="ignore"),
        }


def build_chunks(
    max_docs: int | None,
    local_dir: Path | None,
    chunk_size: int,
    overlap: int,
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_revision: str | None = None,
) -> list[dict]:
    settings = load_settings()
    records = list(
        iter_eurlex_records(
            dataset_name or settings.hf_dataset_name,
            dataset_revision if dataset_revision is not None else settings.hf_dataset_revision,
            max_docs,
            config_name=dataset_config if dataset_name or dataset_config is not None else settings.hf_dataset_config,
        )
    )
    if local_dir:
        records.extend(iter_local_files(local_dir))

    out: list[dict] = []
    for record in tqdm(records, desc="chunking"):
        for i, chunk in enumerate(chunk_text(record["text"], chunk_size, overlap)):
            out.append(
                {
                    "id": f"{record['doc_id']}::chunk{i}",
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "chunk_index": i,
                    "text": chunk,
                }
            )
    return out


def _content_keywords(text: str, n: int = 5) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for raw in WORD_RE.findall(text):
        token = raw.lower()
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        words.append(raw.lower())
        if len(words) >= n:
            break
    return words


def _known_item_query(chunk: dict) -> str:
    text = chunk.get("text") or ""
    mention = ARTICLE_MENTION_RE.search(text)
    title = (chunk.get("title") or "").split(",")[0].strip()
    title_tokens = {token.lower() for token in WORD_RE.findall(title)}
    keywords = [word for word in _content_keywords(text, n=8) if word not in title_tokens][:4]
    focus = ", ".join(keywords) if keywords else "the operative rules"
    if mention and title:
        return f"What does {mention.group(0)} require for {title} regarding {focus}?"
    if mention:
        return f"What does {mention.group(0)} require regarding {focus}?"
    if title:
        return f"Which provisions on {title} address {focus}?"
    return f"Which provision addresses {focus}?"


def _curated_hits(chunks: list[dict], spec: dict, max_relevant: int = 3) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        hay = (chunk.get("text") or "").lower()
        if any(token not in hay for token in spec["all_of"]):
            continue
        hits = sum(1 for token in spec["any_of"] if token in hay)
        if hits <= 0:
            continue
        scored.append((hits, -index, chunk["id"]))
    scored.sort(reverse=True)
    return [chunk_id for _, _, chunk_id in scored[:max_relevant]]


def build_eval_queries(chunks: list[dict], limit: int = 12) -> list[dict]:
    """
    Build a labeled query file for Precision@K and MRR.

    Curated questions are kept only when their anchors occur in this chunk
    set, so gold ids always point at passages that were actually indexed.
    Remaining slots are known-item queries: one distinctive question per
    document, with that chunk id as the single relevant hit.
    """
    if not chunks or limit <= 0:
        return []

    queries: list[dict] = []
    used_texts: set[str] = set()

    for spec in CURATED_QUERIES:
        if len(queries) >= limit:
            break
        relevant = _curated_hits(chunks, spec)
        if not relevant:
            continue
        queries.append({"query": spec["query"], "relevant_chunk_ids": relevant})
        used_texts.add(spec["query"])

    ranked = sorted(
        enumerate(chunks),
        key=lambda pair: (
            0 if ARTICLE_MENTION_RE.search(pair[1].get("text") or "") else 1,
            pair[0],
        ),
    )
    seen_docs: set[str] = set()
    for _, chunk in ranked:
        if len(queries) >= limit:
            break
        if len(chunk.get("text") or "") < 180:
            continue
        doc_id = str(chunk.get("doc_id") or chunk["id"])
        if doc_id in seen_docs:
            continue
        question = _known_item_query(chunk)
        if question in used_texts:
            continue
        seen_docs.add(doc_id)
        used_texts.add(question)
        queries.append({"query": question, "relevant_chunk_ids": [chunk["id"]]})

    if not queries:
        first = chunks[0]
        queries.append({"query": _known_item_query(first), "relevant_chunk_ids": [first["id"]]})
    return queries


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunks_file(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and chunk EU regulatory documents.")
    parser.add_argument("--max-docs", type=int, default=200, help="Cap on source documents.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Optional directory of local .txt/.md regulatory texts to include.",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=Path("data/eval_queries.jsonl"),
        help="Labeled queries aligned to the chunks written by this run.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=12,
        help="How many labeled queries to write.",
    )
    parser.add_argument(
        "--from-chunks",
        type=Path,
        default=None,
        help="Skip the dataset download and build eval_queries.jsonl from an existing chunks file.",
    )
    args = parser.parse_args()

    if args.from_chunks is not None:
        chunks = load_chunks_file(args.from_chunks)
        logger.info("Loaded %d existing chunks from %s", len(chunks), args.from_chunks)
    else:
        chunks = build_chunks(args.max_docs, args.local_dir, args.chunk_size, args.overlap)
        write_jsonl(args.out, chunks)
        logger.info("Wrote %d chunks to %s", len(chunks), args.out)

    queries = build_eval_queries(chunks, limit=args.eval_size)
    write_jsonl(args.eval_out, queries)
    logger.info("Wrote %d labeled queries to %s", len(queries), args.eval_out)


if __name__ == "__main__":
    main()
