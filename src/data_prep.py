"""
Dataset preparation for EU regulatory / legal directives.

Loads the EUR-Lex dataset from Hugging Face (default: joelniklaus/eurlex),
plus optionally any locally supplied .txt/.md files (e.g. GDPR, UK DPA, EU AI
Act text you've legally obtained), and chunks them into overlapping passages
suitable for embedding.

Usage:
    python -m src.data_prep --max-docs 500 --out data/chunks.jsonl
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


def iter_eurlex_records(
    dataset_name: str, revision: str, max_docs: int | None
) -> Iterator[dict]:
    logger.info("Loading dataset %s (revision=%s)...", dataset_name, revision)
    ds = load_dataset(dataset_name, revision=revision, split="train", trust_remote_code=False)
    count = 0
    for row in ds:
        if max_docs is not None and count >= max_docs:
            break
        text = row.get("text") or row.get("celex_text") or ""
        if not text:
            continue
        yield {
            "doc_id": str(row.get("celex_id") or row.get("id") or count),
            "title": row.get("title", "")[:300],
            "text": text,
        }
        count += 1


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
    max_docs: int | None, local_dir: Path | None, chunk_size: int, overlap: int
) -> list[dict]:
    settings = load_settings()
    records = list(
        iter_eurlex_records(settings.hf_dataset_name, settings.hf_dataset_revision, max_docs)
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
    args = parser.parse_args()

    chunks = build_chunks(args.max_docs, args.local_dir, args.chunk_size, args.overlap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Wrote %d chunks to %s", len(chunks), args.out)


if __name__ == "__main__":
    main()
