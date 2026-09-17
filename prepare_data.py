"""
prepare_data.py
----------------
Member 1: Dataset + Baseline Retrieval

Downloads a domain-specific BEIR dataset (corpus + queries + relevance
judgments), cleans the document text, splits it into retrieval-sized
chunks, and saves everything needed for the next steps:

  data/raw/<dataset>/            -> untouched downloaded files
  data/processed/chunks.json     -> cleaned, chunked corpus
  data/processed/queries_qrels.json -> raw queries + relevance labels
                                        (used later to build qa_dataset.json)

Run:
    python prepare_data.py
"""

import json
import re
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ---------------------------------------------------------------------
# Config — change these if you switch datasets or chunking strategy
# ---------------------------------------------------------------------
DATASET = "nfcorpus"          # BEIR dataset name, e.g. "nfcorpus", "scifact", "fiqa"
SPLIT = "test"                 # which split to load (most BEIR sets: train/dev/test)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNK_SIZE_WORDS = 200          # words per chunk
CHUNK_OVERLAP_WORDS = 40        # overlap between consecutive chunks


def download_dataset(dataset: str, out_dir: Path) -> Path:
    """Download and unzip a BEIR dataset, return the path to its folder."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = util.download_and_unzip(url, str(out_dir))
    return Path(data_path)


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, strip weird characters."""
    text = re.sub(r"\s+", " ", text)          # collapse newlines/tabs/multi-spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # drop non-ASCII junk (optional, keep if corpus is English)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping some overlap
    return chunks


def build_chunks(corpus: dict) -> list:
    """Turn the raw BEIR corpus into a flat list of cleaned, chunked passages."""
    all_chunks = []
    for doc_id, doc in corpus.items():
        title = doc.get("title", "")
        body = doc.get("text", "")
        full_text = clean_text(f"{title}. {body}" if title else body)

        pieces = chunk_text(full_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "text": piece,
            })
    return all_chunks


def main():
    print(f"Downloading '{DATASET}' from BEIR...")
    data_path = download_dataset(DATASET, RAW_DIR)

    print("Loading corpus, queries, and relevance labels...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=SPLIT)
    print(f"  {len(corpus)} documents, {len(queries)} queries")

    print("Cleaning and chunking documents...")
    chunks = build_chunks(corpus)
    print(f"  {len(chunks)} chunks created")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    # Save raw queries + qrels so the qa_dataset.json builder (next script)
    # doesn't need to re-download or re-load BEIR.
    with open(PROCESSED_DIR / "queries_qrels.json", "w", encoding="utf-8") as f:
        json.dump({"queries": queries, "qrels": qrels}, f, indent=2)

    print(f"Done. Files saved in {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()