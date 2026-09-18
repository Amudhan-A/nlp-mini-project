import json
from pathlib import Path

from dense_retrieval import DenseRetriever


QA_DATASET = Path("data/qa_dataset.json")


def load_questions():
    with open(QA_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data.get("data", [])

    questions = []

    for r in records:
        question = r.get("question") or r.get("query")
        gold = (
            r.get("gold_answer")
            or r.get("answer")
            or r.get("gold")
        )

        if question and gold:
            questions.append((question, gold, r))

    return questions


def main():

    retriever = DenseRetriever()
    retriever.build_index()

    questions = load_questions()

    print("=" * 70)
    print("RETRIEVAL → ANSWER DIAGNOSTIC")
    print("=" * 70)

    # Inspect 10 questions
    for i, (question, gold, record) in enumerate(questions[:10], 1):

        hits = retriever.search(question, k=5)

        print("\n" + "=" * 70)
        print(f"QUESTION {i}")
        print("=" * 70)

        print(f"\nQUESTION:\n{question}")

        print(f"\nGOLD ANSWER:\n{gold}")

        print("\nRETRIEVED PASSAGES:")

        for j, hit in enumerate(hits, 1):

            print("\n" + "-" * 70)
            print(f"PASSAGE {j}")
            print(f"Doc ID:   {hit['doc_id']}")
            print(f"Chunk ID: {hit['chunk_id']}")

            text = hit["text"]

            # Show enough context without flooding terminal
            print(f"\n{text[:2000]}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()