import json
from pathlib import Path

RESULTS = Path("results")


def load(name):
    path = RESULTS / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_value(obj, possible_keys):
    """Recursively find the first matching key."""
    if isinstance(obj, dict):
        for key in possible_keys:
            if key in obj:
                return obj[key]

        for value in obj.values():
            result = find_value(value, possible_keys)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_value(item, possible_keys)
            if result is not None:
                return result

    return None


def metric(data, *keys):
    return find_value(data, list(keys))


def main():

    bm25 = load("baseline_results.json")
    dense = load("dense_retrieval_results.json")
    distilbert = load("transformer_qa_results.json")
    llama = load("llama_qa_results.json")

    rows = []

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    rows.append({
        "system": "BM25 baseline",
        "recall_at_1": metric(bm25, "recall@1", "recall_at_1"),
        "recall_at_3": metric(bm25, "recall@3", "recall_at_3"),
        "recall_at_5": metric(bm25, "recall@5", "recall_at_5"),
        "recall_at_10": metric(
            bm25,
            "recall@10",
            "recall_at_10"
        ),
        "mrr_at_10": metric(
            bm25,
            "mrr@10",
            "mrr_at_10"
        ),
        "ndcg_at_10": metric(
            bm25,
            "ndcg@10",
            "ndcg_at_10"
        ),
        "exact_match": metric(
            bm25,
            "em",
            "exact_match"
        ),
        "f1": metric(bm25, "f1"),
        "latency_ms": metric(
            bm25,
            "latency_ms_mean",
            "mean_latency_ms",
            "latency_ms"
        ),
    })

    # ---------------------------------------------------------
    # BGE Dense Retrieval
    # ---------------------------------------------------------

    rows.append({
        "system": "BGE dense retrieval",
        "recall_at_1": metric(dense, "recall@1", "recall_at_1"),
        "recall_at_3": metric(dense, "recall@3", "recall_at_3"),
        "recall_at_5": metric(dense, "recall@5", "recall_at_5"),
        "recall_at_10": metric(
            dense,
            "recall@10",
            "recall_at_10"
        ),
        "mrr_at_10": metric(
            dense,
            "mrr@10",
            "mrr_at_10"
        ),
        "ndcg_at_10": metric(
            dense,
            "ndcg@10",
            "ndcg_at_10"
        ),
        "exact_match": None,
        "f1": None,
        "latency_ms": metric(
            dense,
            "latency_ms_mean",
            "mean_latency_ms",
            "latency_ms"
        ),
    })

    # ---------------------------------------------------------
    # BGE + DistilBERT
    # ---------------------------------------------------------

    rows.append({
        "system": "BGE + DistilBERT",
        "recall_at_10": None,
        "mrr_at_10": None,
        "ndcg_at_10": None,
        "exact_match": metric(
            distilbert,
            "exact_match",
            "em"
        ),
        "f1": metric(distilbert, "f1"),
        "latency_ms": metric(
            distilbert,
            "reader_latency_ms_mean",
            "latency_ms_mean",
            "mean_latency_ms",
            "latency_ms"
        ),
    })

    # ---------------------------------------------------------
    # BGE + Llama
    # ---------------------------------------------------------

    rows.append({
        "system": "BGE + Llama 3.1",
        "recall_at_10": None,
        "mrr_at_10": None,
        "ndcg_at_10": None,
        "exact_match": metric(
            llama,
            "exact_match",
            "em"
        ),
        "f1": metric(llama, "f1"),
        "latency_ms": metric(
            llama,
            "latency_ms_mean",
            "mean_latency_ms",
            "latency_ms"
        ),
    })

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL NLP QA SYSTEM COMPARISON")
    print("=" * 90)

    print(
        f"{'System':<24}"
        f"{'R@10':<10}"
        f"{'MRR@10':<10}"
        f"{'nDCG@10':<10}"
        f"{'EM':<10}"
        f"{'F1':<10}"
        f"{'Latency':<12}"
    )

    print("-" * 90)

    for r in rows:

        def fmt(value):
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        print(
            f"{r['system']:<24}"
            f"{fmt(r['recall_at_10']):<10}"
            f"{fmt(r['mrr_at_10']):<10}"
            f"{fmt(r['ndcg_at_10']):<10}"
            f"{fmt(r['exact_match']):<10}"
            f"{fmt(r['f1']):<10}"
            f"{fmt(r['latency_ms']):<12}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = {
        "dataset": "NFCorpus",
        "systems": rows,
        "notes": [
            "BM25 and BGE retrieval metrics evaluate document retrieval.",
            "EM and F1 evaluate generated/extracted answers.",
            "Llama 3.1 runs locally through Ollama.",
            "No external API or API token is required."
        ]
    }

    output_file = RESULTS / "final_comparison.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()