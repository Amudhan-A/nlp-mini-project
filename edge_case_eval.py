"""
edge_case_eval.py
------------------
Runs the 10 hand-written adversarial queries (empty, whitespace, out-of-domain,
code-mixed, OOV, ambiguous, very long, unseen information, non-linguistic)
through the dense retriever + DistilBERT reader, for parity with the
baseline's edge-case logging in results/baseline_results.json.

Requires the Chroma index to already exist (run dense_retrieval.py first).

Usage:
    python edge_case_eval.py
"""

import json
import time
from pathlib import Path

from dense_retrieval import DenseRetriever

QA_FILE = Path("data/qa_dataset.json")
OUT_FILE = Path("results/dense_edge_cases.json")


def main():
    with open(QA_FILE, encoding="utf-8") as f:
        records = json.load(f)

    edge_cases = [r for r in records if r.get("source") == "handwritten"]
    print(f"Found {len(edge_cases)} hand-written edge cases.")

    retriever = DenseRetriever()
    retriever.build_index()

    from qa_reader import TransformerQA
    reader = TransformerQA()

    rows = []
    for rec in edge_cases:
        q = rec["question"]
        t0 = time.perf_counter()
        crashed, error = False, None
        hits, answer_out = [], None
        try:
            hits = retriever.search(q, k=5)
            answer_out = reader.answer(q, hits) if hits else {
                "answer": "", "confidence": -1, "source": None}
        except Exception as exc:
            crashed = True
            error = str(exc)
        latency_ms = (time.perf_counter() - t0) * 1000

        shown = q if len(q) <= 60 else q[:60] + "..."
        ans_preview = (answer_out or {}).get("answer", "")[:60]
        print(f"[{rec.get('category', '?'):<22}] {shown!r} "
              f"-> {len(hits)} hits, ans={ans_preview!r}, {latency_ms:.1f} ms"
              + (f"  CRASHED: {error}" if crashed else ""))

        rows.append({
            "id": rec.get("id"),
            "category": rec.get("category"),
            "question": q,
            "expected_behaviour": rec.get("expected_behaviour"),
            "n_hits": len(hits),
            "top_doc_id": hits[0]["doc_id"] if hits else None,
            "top_distance": hits[0].get("distance") if hits else None,
            "predicted_answer": (answer_out or {}).get("answer"),
            "answer_confidence": (answer_out or {}).get("confidence"),
            "latency_ms": round(latency_ms, 2),
            "crashed": crashed,
            "error": error,
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    n_crashed = sum(1 for r in rows if r["crashed"])
    print(f"\n{n_crashed}/{len(rows)} edge cases raised an exception.")
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()