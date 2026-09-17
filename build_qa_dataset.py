"""
build_qa_dataset.py
-------------------
Member 1: Dataset + Baseline Retrieval  (step 2 of 3)

Turns the raw BEIR queries + relevance judgments produced by prepare_data.py
into the single evaluation file the whole team consumes:

    data/qa_dataset.json

What it does
------------
1. Loads data/processed/queries_qrels.json and data/processed/chunks.json
2. Keeps only queries that have at least one relevant document present in
   the corpus (some BEIR qrels point at docs outside the split)
3. Maps each relevant doc_id -> the chunk_ids that came from it
4. Fixes a reproducible train / validation / test split (seed = 42)
5. Flags a subset of questions for manual gold-answer annotation
   (BEIR ships relevance labels, NOT answer strings -- Exact Match and F1
   need answer text, so a human has to write those for a small subset)
6. Appends hand-written edge-case questions the assignment explicitly asks
   for: empty input, out-of-domain, very long, ambiguous, code-mixed,
   unseen information

Run:
    python build_qa_dataset.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_FILE = Path("data/qa_dataset.json")

RANDOM_SEED = 42            # assignment asks for a stated random seed
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test
N_TO_ANNOTATE = 30          # how many questions to flag for gold answers
MIN_RELEVANCE = 1           # qrel score > this is "relevant" (NFCorpus is graded 0/1/2)


# ---------------------------------------------------------------------
# Hand-written questions the stock dataset cannot give you.
# Edit these to match whichever corpus you ended up using.
# `gold_answer` may be null where the correct behaviour is a refusal.
# ---------------------------------------------------------------------
EDGE_CASE_QUESTIONS = [
    {
        "question": "",
        "category": "empty_input",
        "expected_behaviour": "Reject politely: no question to process.",
        "gold_answer": None,
    },
    {
        "question": "   ",
        "category": "whitespace_only_input",
        "expected_behaviour": "Treated the same as empty input.",
        "gold_answer": None,
    },
    {
        "question": "Who won the 2018 FIFA World Cup final?",
        "category": "out_of_domain",
        "expected_behaviour": "Refuse: outside the knowledge base.",
        "gold_answer": None,
    },
    {
        "question": "What is the capital city of Australia?",
        "category": "out_of_domain",
        "expected_behaviour": "Refuse: outside the knowledge base.",
        "gold_answer": None,
    },
    {
        "question": "vitamin D deficiency epaddi affect pannum?",
        "category": "code_mixed",
        "expected_behaviour": "Retrieve on the English content words; do not crash.",
        "gold_answer": None,
    },
    {
        "question": "Does xylomethaquinolase supplementation reduce tumour growth?",
        "category": "out_of_vocabulary",
        "expected_behaviour": "Low retrieval score; system should say it is unsure.",
        "gold_answer": None,
    },
    {
        "question": "Is it good?",
        "category": "ambiguous",
        "expected_behaviour": "Ask for clarification or return a low-confidence answer.",
        "gold_answer": None,
    },
    {
        "question": (
            "I have been reading a great deal recently about nutrition and chronic disease, "
            "and I keep coming across conflicting claims, so I wanted to ask a fairly detailed "
            "question that covers several things at once. " * 30
        ).strip(),
        "category": "very_long_input",
        "expected_behaviour": "Truncate to the model window; still return a result.",
        "gold_answer": None,
    },
    {
        "question": "What were the findings of the 2027 multi-country trial on this topic?",
        "category": "unseen_information",
        "expected_behaviour": "Refuse: information not in the corpus.",
        "gold_answer": None,
    },
    {
        "question": "!!!! ???? ####",
        "category": "non_linguistic_input",
        "expected_behaviour": "No usable query terms; refuse gracefully.",
        "gold_answer": None,
    },
]


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_doc_to_chunks(chunks: list) -> dict:
    """doc_id -> [chunk_id, ...] in original order."""
    mapping = defaultdict(list)
    for c in chunks:
        mapping[c["doc_id"]].append(c["chunk_id"])
    return mapping


def build_records(queries: dict, qrels: dict, doc_to_chunks: dict) -> list:
    """One record per usable BEIR query."""
    records = []
    skipped = 0

    for qid, question in queries.items():
        judged = qrels.get(qid, {})

        relevant_docs, graded = [], {}
        for doc_id, score in judged.items():
            if score >= MIN_RELEVANCE and doc_id in doc_to_chunks:
                relevant_docs.append(doc_id)
                graded[doc_id] = int(score)

        if not relevant_docs:
            skipped += 1
            continue

        relevant_chunks = []
        for doc_id in relevant_docs:
            relevant_chunks.extend(doc_to_chunks[doc_id])

        records.append({
            "id": qid,
            "question": question.strip(),
            "source": "beir",
            "category": "normal",
            "relevant_doc_ids": relevant_docs,
            "relevant_chunk_ids": relevant_chunks,
            "relevance_grades": graded,
            "gold_answer": None,
            "needs_gold_answer": False,
            "expected_behaviour": None,
        })

    print(f"  kept {len(records)} queries, skipped {skipped} with no in-corpus relevant doc")
    return records


def assign_splits(records: list, ratios, seed: int):
    """Deterministic train/val/test assignment."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    for i, rec in enumerate(shuffled):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_val:
            rec["split"] = "validation"
        else:
            rec["split"] = "test"
    return shuffled


def flag_for_annotation(records: list, n: int, seed: int):
    """
    Pick n test-split questions for manual gold-answer writing.
    Prefers questions whose shortest relevant doc is small -- easier to
    read and annotate by hand.
    """
    rng = random.Random(seed)
    test_records = [r for r in records if r["split"] == "test"]
    chosen = rng.sample(test_records, min(n, len(test_records)))
    for rec in chosen:
        rec["needs_gold_answer"] = True
    return len(chosen)


def main():
    qq = load_json(PROCESSED_DIR / "queries_qrels.json")
    chunks = load_json(PROCESSED_DIR / "chunks.json")

    queries, qrels = qq["queries"], qq["qrels"]
    print(f"Loaded {len(queries)} queries, {len(chunks)} chunks")

    doc_to_chunks = build_doc_to_chunks(chunks)
    records = build_records(queries, qrels, doc_to_chunks)
    records = assign_splits(records, SPLIT_RATIOS, RANDOM_SEED)

    n_flagged = flag_for_annotation(records, N_TO_ANNOTATE, RANDOM_SEED)
    print(f"  flagged {n_flagged} test questions for manual gold answers")

    # Edge cases: always test split, never part of retrieval scoring
    for i, ec in enumerate(EDGE_CASE_QUESTIONS):
        records.append({
            "id": f"edge_{i:02d}",
            "question": ec["question"],
            "source": "handwritten",
            "category": ec["category"],
            "relevant_doc_ids": [],
            "relevant_chunk_ids": [],
            "relevance_grades": {},
            "gold_answer": ec["gold_answer"],
            "needs_gold_answer": False,
            "expected_behaviour": ec["expected_behaviour"],
            "split": "test",
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    counts = defaultdict(int)
    for r in records:
        counts[r["split"]] += 1

    print(f"\nWrote {len(records)} records to {OUT_FILE}")
    print(f"  train={counts['train']}  validation={counts['validation']}  test={counts['test']}")
    print(f"  edge cases: {len(EDGE_CASE_QUESTIONS)}")
    print(f"  random seed: {RANDOM_SEED}")
    print(
        f"\nNEXT: open {OUT_FILE}, find the {n_flagged} records with "
        '"needs_gold_answer": true, and fill in "gold_answer" by reading the '
        "relevant document. Those are the only ones Exact Match / F1 can score."
    )


if __name__ == "__main__":
    main()