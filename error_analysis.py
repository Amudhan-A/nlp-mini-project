"""
Generate a 10-case error-analysis report from the saved Llama evaluation.

The script selects the lowest-F1 questions, retrieves the same top-5 dense
passages, and records the question, gold answer, model answer, and evidence.
It does not call Llama again.
"""

import json
from pathlib import Path

from dense_retrieval import DenseRetriever

QA_FILE = Path("data/qa_dataset.json")
LLAMA_FILE = Path("results/llama_qa_results.json")
OUT_JSON = Path("results/error_analysis_10.json")
OUT_MD = Path("results/error_analysis_10.md")


def main():
    with open(LLAMA_FILE, "r", encoding="utf-8") as f:
        llama = json.load(f)

    cases = sorted(
        llama["per_question"],
        key=lambda x: (x.get("f1", 0), x.get("em", 0))
    )[:10]

    retriever = DenseRetriever()
    output = []

    for rank, case in enumerate(cases, 1):
        passages = retriever.search(case["question"], k=5)

        output.append({
            "case": rank,
            "question": case["question"],
            "gold_answer": case["gold_answer"],
            "predicted_answer": case["predicted_answer"],
            "em": case["em"],
            "f1": case["f1"],
            "retrieved_evidence": [
                {
                    "rank": i,
                    "doc_id": p.get("doc_id", "unknown"),
                    "text": p.get("text", "")
                }
                for i, p in enumerate(passages, 1)
            ]
        })

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    lines = [
        "# Error Analysis — 10 Llama 3.1 QA Failures",
        "",
        "The ten cases below are the lowest-F1 cases from the saved 30-question evaluation.",
        "The analysis separates retrieval problems from answer-generation/reference-matching problems.",
        "",
    ]

    for c in output:
        lines += [
            f"## Case {c['case']}: {c['question']}",
            f"- **EM:** {c['em']:.4f}",
            f"- **F1:** {c['f1']:.4f}",
            f"- **Gold answer:** {c['gold_answer']}",
            f"- **Predicted answer:** {c['predicted_answer']}",
            "",
            "### Retrieved evidence",
        ]
        for e in c["retrieved_evidence"]:
            text = " ".join(e["text"].split())
            if len(text) > 900:
                text = text[:900] + "..."
            lines += [f"**[{e['rank']}] {e['doc_id']}** — {text}", ""]

        lines += [
            "### Failure interpretation",
            "Inspect whether the retrieved passages contain the information needed for the gold answer.",
            "If the evidence is absent or weak, classify primarily as a **retrieval failure**.",
            "If relevant evidence is present but the answer is incomplete, overly broad, or differently worded, classify primarily as a **reader/generation or reference-matching failure**.",
            "",
        ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved report: {OUT_MD}")


if __name__ == "__main__":
    main()
