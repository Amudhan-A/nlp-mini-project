"""
baseline.py
-----------
Member 1: Dataset + Baseline Retrieval  (step 3 of 3)

The classical, non-neural baseline the transformer system gets compared
against. Contains three things:

  1. A retriever (TF-IDF cosine  OR  BM25) over data/processed/chunks.json
  2. A crude extractive "answer" step: pick the sentence in the top chunk
     with the most overlap with the question. No neural model involved --
     that is the point, it is the floor Member 2 has to beat.
  3. Evaluation: Recall@k, MRR@10, nDCG@10, Exact Match and token F1,
     plus latency and index-memory numbers the assignment asks for.

Usage
-----
    python baseline.py                      # evaluate on the test split
    python baseline.py --method tfidf       # switch scorer (default: bm25)
    python baseline.py --split validation
    python baseline.py --edge-cases         # run the edge-case demo only
    python baseline.py --ask "your question here"

Importable by the rest of the team:
    from baseline import BaselineRetriever
    r = BaselineRetriever.from_files(method="bm25")
    hits = r.search("does vitamin D reduce cancer risk?", k=5)
"""

import argparse
import json
import math
import re
import statistics
import string
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CHUNKS_FILE = Path("data/processed/chunks.json")
QA_FILE = Path("data/qa_dataset.json")
RESULTS_FILE = Path("results/baseline_results.json")

DEFAULT_METHOD = "bm25"        # "bm25" or "tfidf"
TOP_K = 10
K_VALUES = (1, 3, 5, 10)
MAX_QUESTION_WORDS = 400       # anything longer gets truncated
MIN_SCORE = 0.05               # normalised score below this => "I don't know"


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles/punctuation/extra space."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------
class BaselineRetriever:
    """Classical lexical retrieval over the chunked corpus."""

    def __init__(self, chunks: list, method: str = DEFAULT_METHOD):
        if method not in ("bm25", "tfidf"):
            raise ValueError("method must be 'bm25' or 'tfidf'")
        self.method = method
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        tracemalloc.start()
        t0 = time.perf_counter()

        if method == "bm25":
            from rank_bm25 import BM25Okapi
            self.tokenized = [tokenize(t) for t in self.texts]
            self.model = BM25Okapi(self.tokenized)
            self.vectorizer = None
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(self.texts)
            self.model = None

        self.build_time_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.index_memory_mb = peak / (1024 ** 2)

    @classmethod
    def from_files(cls, chunks_file: Path = CHUNKS_FILE, method: str = DEFAULT_METHOD):
        with open(chunks_file, encoding="utf-8") as f:
            chunks = json.load(f)
        return cls(chunks, method=method)

    # -- query handling ------------------------------------------------
    def _prepare_query(self, question: str):
        """Returns (cleaned_question, note) or (None, reason) if unusable."""
        if question is None:
            return None, "empty_input"
        q = question.strip()
        if not q:
            return None, "empty_input"

        words = q.split()
        note = None
        if len(words) > MAX_QUESTION_WORDS:
            q = " ".join(words[:MAX_QUESTION_WORDS])
            note = f"truncated from {len(words)} to {MAX_QUESTION_WORDS} words"

        if not tokenize(q):
            return None, "no_usable_terms"
        return q, note

    def search(self, question: str, k: int = TOP_K) -> list:
        """Top-k chunks as [{chunk_id, doc_id, text, score, norm_score}]."""
        q, note = self._prepare_query(question)
        if q is None:
            return []

        if self.method == "bm25":
            scores = np.asarray(self.model.get_scores(tokenize(q)))
        else:
            qv = self.vectorizer.transform([q])
            scores = (self.matrix @ qv.T).toarray().ravel()

        top_idx = np.argsort(-scores)[:k]
        best = float(scores[top_idx[0]]) if len(top_idx) else 0.0
        denom = best if best > 0 else 1.0

        hits = []
        for i in top_idx:
            c = self.chunks[i]
            hits.append({
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "score": float(scores[i]),
                # relative to the best hit for this query -- used as a
                # cheap confidence signal for the out-of-domain check
                "norm_score": float(scores[i]) / denom,
            })
        if note and hits:
            hits[0]["note"] = note
        return hits

    # -- crude extractive answer --------------------------------------
    def answer(self, question: str, k: int = 5) -> dict:
        """
        Full baseline QA: retrieve, then return the sentence from the best
        chunk that shares the most content words with the question.
        """
        q, note = self._prepare_query(question)
        if q is None:
            reason = ("I don't have a question to process."
                      if note == "empty_input"
                      else "I couldn't find any usable search terms in that input.")
            return {"answer": reason, "confidence": 0.0, "sources": [],
                    "status": note, "hits": []}

        hits = self.search(q, k=k)
        if not hits or hits[0]["score"] <= 0:
            return {"answer": "I couldn't find sufficient information to answer that.",
                    "confidence": 0.0, "sources": [], "status": "no_match", "hits": []}

        # absolute-score guard for obviously out-of-domain questions:
        # BM25/TF-IDF scores are unbounded, so use overlap as the signal
        q_toks = set(tokenize(q))
        top_toks = set(tokenize(hits[0]["text"]))
        overlap = len(q_toks & top_toks) / max(len(q_toks), 1)

        if overlap < MIN_SCORE:
            return {"answer": "Sorry, this question looks to be outside my knowledge base.",
                    "confidence": round(overlap, 3), "sources": [], "status": "out_of_domain",
                    "hits": hits}

        sentences = split_sentences(hits[0]["text"]) or [hits[0]["text"]]
        best_sent, best_score = sentences[0], -1.0
        for sent in sentences:
            s_toks = set(tokenize(sent))
            if not s_toks:
                continue
            score = len(q_toks & s_toks) / math.sqrt(len(s_toks))
            if score > best_score:
                best_sent, best_score = sent, score

        return {
            "answer": best_sent,
            "confidence": round(overlap, 3),
            "sources": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                         "score": round(h["score"], 4)} for h in hits],
            "status": "truncated" if note else "ok",
            "hits": hits,
        }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def ranked_doc_ids(hits: list) -> list:
    """Chunk ranking -> deduplicated document ranking, order preserved."""
    seen, docs = set(), []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.add(h["doc_id"])
            docs.append(h["doc_id"])
    return docs


def ndcg_at_k(ranked_docs: list, grades: dict, k: int = 10) -> float:
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked_docs[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(retriever: BaselineRetriever, records: list, k_values=K_VALUES) -> dict:
    scored = [r for r in records if r["relevant_doc_ids"]]
    recalls = {k: [] for k in k_values}
    rr, ndcgs, latencies = [], [], []
    ems, f1s = [], []
    per_question = []

    for rec in scored:
        t0 = time.perf_counter()
        hits = retriever.search(rec["question"], k=max(k_values))
        latencies.append((time.perf_counter() - t0) * 1000)

        docs = ranked_doc_ids(hits)
        gold = set(rec["relevant_doc_ids"])

        for k in k_values:
            recalls[k].append(float(any(d in gold for d in docs[:k])))

        rank = next((i + 1 for i, d in enumerate(docs[:10]) if d in gold), None)
        rr.append(1.0 / rank if rank else 0.0)
        grades = {d: int(g) for d, g in rec["relevance_grades"].items()}
        ndcgs.append(ndcg_at_k(docs, grades, k=10))

        entry = {
            "id": rec["id"],
            "question": rec["question"][:120],
            "gold_docs": rec["relevant_doc_ids"][:5],
            "retrieved_docs": docs[:5],
            "hit_rank": rank,
            "ndcg@10": round(ndcgs[-1], 4),
        }

        if rec.get("gold_answer"):
            pred = retriever.answer(rec["question"])["answer"]
            em, f1 = exact_match(pred, rec["gold_answer"]), token_f1(pred, rec["gold_answer"])
            ems.append(em)
            f1s.append(f1)
            entry.update({"predicted_answer": pred[:200],
                          "gold_answer": rec["gold_answer"],
                          "em": em, "f1": round(f1, 4)})

        per_question.append(entry)

    metrics = {
        "method": retriever.method,
        "n_questions_scored": len(scored),
        "n_with_gold_answers": len(ems),
        **{f"recall@{k}": round(float(np.mean(recalls[k])), 4) for k in k_values},
        "mrr@10": round(float(np.mean(rr)), 4) if rr else 0.0,
        "ndcg@10": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "exact_match": round(float(np.mean(ems)), 4) if ems else None,
        "f1": round(float(np.mean(f1s)), 4) if f1s else None,
        "latency_ms_mean": round(statistics.mean(latencies), 2),
        "latency_ms_median": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2),
        "index_build_time_s": round(retriever.build_time_s, 2),
        "index_peak_memory_mb": round(retriever.index_memory_mb, 2),
    }
    return {"metrics": metrics, "per_question": per_question}


# ---------------------------------------------------------------------
# Edge-case demo
# ---------------------------------------------------------------------
def run_edge_cases(retriever: BaselineRetriever, records: list) -> list:
    edge = [r for r in records if r["source"] == "handwritten"]
    rows = []
    print("\n--- Edge cases -------------------------------------------")
    for rec in edge:
        t0 = time.perf_counter()
        out = retriever.answer(rec["question"])
        ms = (time.perf_counter() - t0) * 1000
        shown = (rec["question"][:60] + "...") if len(rec["question"]) > 60 else repr(rec["question"])
        print(f"[{rec['category']:<22}] {shown}")
        print(f"    -> {out['answer'][:110]}")
        print(f"    status={out['status']}  confidence={out['confidence']}  {ms:.1f} ms")
        rows.append({"id": rec["id"], "category": rec["category"],
                     "expected": rec["expected_behaviour"],
                     "status": out["status"], "answer": out["answer"][:200],
                     "confidence": out["confidence"], "latency_ms": round(ms, 2)})
    return rows


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=["bm25", "tfidf"])
    ap.add_argument("--split", default="test",
                    choices=["train", "validation", "test", "all"])
    ap.add_argument("--edge-cases", action="store_true", help="run edge-case demo only")
    ap.add_argument("--ask", type=str, help="answer one question and exit")
    ap.add_argument("--compare", action="store_true", help="evaluate bm25 and tfidf")
    args = ap.parse_args()

    with open(QA_FILE, encoding="utf-8") as f:
        all_records = json.load(f)

    print(f"Building {args.method} index over {CHUNKS_FILE}...")
    retriever = BaselineRetriever.from_files(method=args.method)
    print(f"  {len(retriever.chunks)} chunks indexed in "
          f"{retriever.build_time_s:.2f}s, peak {retriever.index_memory_mb:.1f} MB")

    if args.ask is not None:
        out = retriever.answer(args.ask)
        print(f"\nQ: {args.ask}\nA: {out['answer']}\n"
              f"status={out['status']} confidence={out['confidence']}")
        for s in out["sources"][:3]:
            print(f"   source: {s['doc_id']}  (chunk {s['chunk_id']}, score {s['score']})")
        return

    if args.edge_cases:
        run_edge_cases(retriever, all_records)
        return

    records = ([r for r in all_records if r["split"] == args.split]
               if args.split != "all" else all_records)
    print(f"Evaluating on {args.split} split ({len(records)} records)...")

    methods = ["bm25", "tfidf"] if args.compare else [args.method]
    all_results = {}
    for m in methods:
        r = retriever if m == args.method else BaselineRetriever.from_files(method=m)
        res = evaluate(r, records)
        all_results[m] = res
        print(f"\n--- {m.upper()} on {args.split} ---")
        for key, val in res["metrics"].items():
            print(f"  {key:<24} {val}")

    edge_rows = run_edge_cases(retriever, all_records)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "results": all_results,
                   "edge_cases": edge_rows}, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    if all_results[methods[0]]["metrics"]["exact_match"] is None:
        print("\nNOTE: Exact Match / F1 are null because no gold_answer fields are "
              "filled in yet. Fill the records flagged needs_gold_answer in "
              f"{QA_FILE} and re-run.")


if __name__ == "__main__":
    main()