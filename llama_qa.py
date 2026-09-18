import json
import time
import urllib.request
import urllib.error
import statistics
import re
import string
from collections import Counter
from pathlib import Path

from dense_retrieval import DenseRetriever


MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

QA_FILE = Path("data/qa_dataset.json")
RESULTS_FILE = Path("results/llama_qa_results.json")

TOP_K = 5


# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------

def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(
        normalize_answer(pred) == normalize_answer(gold)
    )


def token_f1(pred, gold):
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()

    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)

    common = Counter(p_tokens) & Counter(g_tokens)
    same = sum(common.values())

    if same == 0:
        return 0.0

    precision = same / len(p_tokens)
    recall = same / len(g_tokens)

    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------
# Llama
# ------------------------------------------------------------

def ask_llama(question, passages):

    context = "\n\n".join(
        f"[Source {i+1}]\n{p.get('text', '')}"
        for i, p in enumerate(passages)
    )

    prompt = f"""You are a question-answering system for a medical and nutrition information database.

Answer the question using ONLY the information in the provided context.

Do not use outside knowledge.
Do not invent facts.
If the context does not contain enough information, say:
"Insufficient information in the retrieved context."

Give a concise factual answer.

Question:
{question}

Context:
{context}

Answer:
"""

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    return result["response"].strip()


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("DENSE RETRIEVAL + LLAMA 3.1 QA EVALUATION")
    print("=" * 60)

    # Load QA dataset
    with open(QA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Only questions with gold answers
    questions = [
        r for r in records
        if r.get("gold_answer")
    ]

    print(f"\nGold-answer questions: {len(questions)}")

    # Load dense retriever
    retriever = DenseRetriever()

    em_scores = []
    f1_scores = []
    latencies = []

    per_question = []

    for idx, rec in enumerate(questions, 1):

        question = rec["question"]
        gold = rec["gold_answer"]

        print("\n" + "-" * 60)
        print(f"[{idx}/{len(questions)}] {question}")

        # Dense retrieval
        retrieval_start = time.perf_counter()

        passages = retriever.search(
            question,
            k=TOP_K
        )

        retrieval_latency = (
            time.perf_counter() - retrieval_start
        ) * 1000

        # Llama generation
        generation_start = time.perf_counter()

        try:
            answer = ask_llama(
                question,
                passages
            )

            generation_latency = (
                time.perf_counter() - generation_start
            ) * 1000

            total_latency = (
                retrieval_latency +
                generation_latency
            )

        except Exception as e:
            answer = f"ERROR: {e}"
            generation_latency = 0
            total_latency = retrieval_latency

        # Metrics
        em = exact_match(answer, gold)
        f1 = token_f1(answer, gold)

        em_scores.append(em)
        f1_scores.append(f1)
        latencies.append(total_latency)

        retrieved_docs = [
            p.get("doc_id", "unknown")
            for p in passages
        ]

        print(f"Answer: {answer}")
        print(f"EM: {em:.4f} | F1: {f1:.4f}")
        print(f"Latency: {total_latency:.2f} ms")

        per_question.append({
            "id": rec.get("id"),
            "question": question,
            "gold_answer": gold,
            "predicted_answer": answer,
            "em": round(em, 4),
            "f1": round(f1, 4),
            "retrieved_docs": retrieved_docs,
            "retrieval_latency_ms": round(
                retrieval_latency, 2
            ),
            "generation_latency_ms": round(
                generation_latency, 2
            ),
            "total_latency_ms": round(
                total_latency, 2
            )
        })

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = {
        "model": MODEL,
        "retrieval_model": "BAAI/bge-small-en-v1.5",
        "top_k": TOP_K,
        "n_questions": len(questions),

        "exact_match": round(
            statistics.mean(em_scores), 4
        ),

        "f1": round(
            statistics.mean(f1_scores), 4
        ),

        "latency_ms_mean": round(
            statistics.mean(latencies), 2
        ),

        "latency_ms_median": round(
            statistics.median(latencies), 2
        ),

        "latency_ms_p95": round(
            sorted(latencies)[
                int(0.95 * len(latencies)) - 1
            ],
            2
        )
    }

    output = {
        "metrics": metrics,
        "per_question": per_question
    }

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 60)
    print("LLAMA 3.1 QA RESULTS")
    print("=" * 60)

    print(f"Model:              {MODEL}")
    print(f"Retrieval model:    BAAI/bge-small-en-v1.5")
    print(f"Questions:          {len(questions)}")
    print(f"Exact Match:        {metrics['exact_match']:.4f}")
    print(f"F1:                 {metrics['f1']:.4f}")
    print(f"Mean latency:       {metrics['latency_ms_mean']:.2f} ms")
    print(f"Median latency:     {metrics['latency_ms_median']:.2f} ms")
    print(f"P95 latency:        {metrics['latency_ms_p95']:.2f} ms")

    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()