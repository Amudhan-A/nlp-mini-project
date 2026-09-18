import json
import math
import time
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNKS_FILE = Path("data/processed/chunks.json")
QA_FILE = Path("data/qa_dataset.json")
RESULTS_FILE = Path("results/dense_retrieval_results.json")

DB_DIR = "data/vector_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "nfcorpus_dense"

K_VALUES = (1, 3, 5, 10)


# ---------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------

class DenseRetriever:

    def __init__(self):

        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading document chunks...")
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"Loaded {len(self.chunks)} chunks.")

        # Local persistent Chroma database
        self.client = chromadb.PersistentClient(path=DB_DIR)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Vector database contains "
            f"{self.collection.count()} chunks."
        )


    # -----------------------------------------------------
    # Build vector database
    # -----------------------------------------------------

    def build_index(self):

        # Prevent accidentally rebuilding an existing index
        if self.collection.count() > 0:
            print("Index already exists. Skipping embedding generation.")
            return

        print("Generating document embeddings...")

        texts = [chunk["text"] for chunk in self.chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print("Storing embeddings in Chroma...")

        # Chroma has a maximum batch size
        BATCH_SIZE = 5000

        for start in range(0, len(self.chunks), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(self.chunks))

            print(f"Storing chunks {start + 1}-{end}...")

            self.collection.upsert(
                ids=[
                    chunk["chunk_id"]
                    for chunk in self.chunks[start:end]
                ],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=[
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"]
                    }
                    for chunk in self.chunks[start:end]
                ]
            )

        print(f"Indexed {len(texts)} chunks successfully.")


    # -----------------------------------------------------
    # Dense Search
    # -----------------------------------------------------

    def search(self, question, k=10):

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )

        hits = []

        if not results["documents"] or not results["documents"][0]:
            return hits

        for i in range(len(results["documents"][0])):

            hits.append({
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i]
            })

        return hits


# ---------------------------------------------------------
# Convert chunk ranking → document ranking
# ---------------------------------------------------------

def ranked_doc_ids(hits):

    seen = set()
    docs = []

    for hit in hits:

        doc_id = hit["doc_id"]

        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)

    return docs


# ---------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------

def recall_at_k(ranked_docs, gold_docs, k):

    retrieved = set(ranked_docs[:k])

    return float(bool(retrieved & gold_docs))


def reciprocal_rank(ranked_docs, gold_docs):

    for i, doc_id in enumerate(ranked_docs[:10], start=1):

        if doc_id in gold_docs:
            return 1.0 / i

    return 0.0


def ndcg_at_k(ranked_docs, relevance_grades, k=10):

    dcg = 0.0

    for i, doc_id in enumerate(ranked_docs[:k]):

        grade = relevance_grades.get(doc_id, 0)

        dcg += (
            (2 ** grade - 1)
            / math.log2(i + 2)
        )

    ideal_grades = sorted(
        relevance_grades.values(),
        reverse=True
    )[:k]

    idcg = 0.0

    for i, grade in enumerate(ideal_grades):

        idcg += (
            (2 ** grade - 1)
            / math.log2(i + 2)
        )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(retriever, records):

    recalls = {
        k: []
        for k in K_VALUES
    }

    reciprocal_ranks = []
    ndcgs = []
    latencies = []

    per_question = []

    for index, record in enumerate(records, start=1):

        question = record["question"]

        gold_docs = set(
            record["relevant_doc_ids"]
        )

        t0 = time.perf_counter()

        hits = retriever.search(
            question,
            k=10
        )

        latency = (
            time.perf_counter() - t0
        ) * 1000

        latencies.append(latency)

        ranked_docs = ranked_doc_ids(hits)

        # Recall@k
        for k in K_VALUES:

            recalls[k].append(
                recall_at_k(
                    ranked_docs,
                    gold_docs,
                    k
                )
            )

        # MRR@10
        rr = reciprocal_rank(
            ranked_docs,
            gold_docs
        )

        reciprocal_ranks.append(rr)

        # nDCG@10
        relevance_grades = {
            doc_id: int(score)
            for doc_id, score
            in record["relevance_grades"].items()
        }

        ndcg = ndcg_at_k(
            ranked_docs,
            relevance_grades,
            k=10
        )

        ndcgs.append(ndcg)

        # Store per-question result
        per_question.append({
            "id": record["id"],
            "question": question,
            "gold_docs": list(gold_docs),
            "retrieved_docs": ranked_docs,
            "hit_rank": (
                int(1 / rr)
                if rr > 0
                else None
            ),
            "recall@1": recalls[1][-1],
            "recall@3": recalls[3][-1],
            "recall@5": recalls[5][-1],
            "recall@10": recalls[10][-1],
            "mrr@10": rr,
            "ndcg@10": ndcg,
            "latency_ms": latency
        })

        print(
            f"[{index:02d}/{len(records)}] "
            f"R@5={recalls[5][-1]:.0f} "
            f"RR={rr:.3f} "
            f"Latency={latency:.1f} ms"
        )


    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    metrics = {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "n_questions": len(records),

        "recall@1": round(
            float(np.mean(recalls[1])),
            4
        ),

        "recall@3": round(
            float(np.mean(recalls[3])),
            4
        ),

        "recall@5": round(
            float(np.mean(recalls[5])),
            4
        ),

        "recall@10": round(
            float(np.mean(recalls[10])),
            4
        ),

        "mrr@10": round(
            float(np.mean(reciprocal_ranks)),
            4
        ),

        "ndcg@10": round(
            float(np.mean(ndcgs)),
            4
        ),

        "latency_ms_mean": round(
            float(np.mean(latencies)),
            2
        ),

        "latency_ms_median": round(
            float(np.median(latencies)),
            2
        )
    }

    return {
        "metrics": metrics,
        "per_question": per_question
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = DenseRetriever()

    # Build only if necessary
    retriever.build_index()

    # Load evaluation dataset
    print("\nLoading QA dataset...")

    with open(QA_FILE, encoding="utf-8") as f:
        records = json.load(f)

    # Same test split used by the baseline
    test_records = [
        r
        for r in records
        if r["split"] == "test"
        and r["relevant_doc_ids"]
    ]

    print(
        f"Evaluating dense retrieval "
        f"on {len(test_records)} test questions..."
    )

    results = evaluate(
        retriever,
        test_records
    )

    print("\n" + "=" * 60)
    print("DENSE RETRIEVAL RESULTS")
    print("=" * 60)

    for key, value in results["metrics"].items():

        print(
            f"{key:<25} {value}"
        )

    # Save results
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
            results,
            f,
            indent=2
        )

    print(
        f"\nResults saved to "
        f"{RESULTS_FILE}"
    )