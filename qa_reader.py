import json
import re
import time
import string
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from dense_retrieval import DenseRetriever


# ============================================================
# CONFIGURATION
# ============================================================

QA_MODEL = "distilbert-base-cased-distilled-squad"
QA_DATASET = Path("data/qa_dataset.json")
RESULT_FILE = Path("results/transformer_qa_results.json")

TOP_K = 5
MAX_LENGTH = 512

# Maximum number of tokens in an extracted answer
MAX_ANSWER_LENGTH = 40

# Number of candidate start/end positions
N_BEST_START = 20
N_BEST_END = 20


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return " ".join(text.split())


def get_question_terms(question):
    """
    Get meaningful words from the question.
    """

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were",
        "do", "does", "did", "of", "to", "for", "in",
        "on", "and", "or", "with", "from", "by", "how",
        "what", "which", "who", "why", "when", "where",
        "should", "can", "could", "would", "their", "they",
        "it", "its", "about"
    }

    words = normalize_text(question).split()

    return {
        word for word in words
        if word not in stopwords and len(word) > 2
    }


def question_overlap(question, answer):
    """
    Measure how many meaningful question terms occur
    in the candidate answer.
    """

    q_terms = get_question_terms(question)

    if not q_terms:
        return 0.0

    answer_terms = set(
        normalize_text(answer).split()
    )

    overlap = len(q_terms & answer_terms)

    return overlap / len(q_terms)


# ============================================================
# EVALUATION METRICS
# ============================================================

def normalize_answer(text):

    text = text.lower()

    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    return " ".join(text.split())


def exact_match(prediction, ground_truth):

    return int(
        normalize_answer(prediction)
        == normalize_answer(ground_truth)
    )


def f1_score(prediction, ground_truth):

    pred_tokens = normalize_answer(
        prediction
    ).split()

    gold_tokens = normalize_answer(
        ground_truth
    ).split()

    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    common = {}

    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0

    for token in gold_tokens:

        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# TRANSFORMER QA
# ============================================================

class TransformerQA:

    def __init__(self):

        print("Loading Transformer QA model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            QA_MODEL
        )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            QA_MODEL
        )

        self.model.eval()

        print("Transformer QA model loaded.")

    # --------------------------------------------------------
    # Extract candidate answers
    # --------------------------------------------------------

    def get_candidates(self, question, context):

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True
        )

        offsets = inputs.pop("offset_mapping")[0]

        sequence_ids = inputs.sequence_ids(0)

        context_indices = [
            i
            for i, sid in enumerate(sequence_ids)
            if sid == 1
        ]

        if not context_indices:
            return []

        context_start = context_indices[0]
        context_end = context_indices[-1]

        with torch.no_grad():

            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_values = start_logits[
            context_start:context_end + 1
        ]

        end_values = end_logits[
            context_start:context_end + 1
        ]

        start_top = torch.topk(
            start_values,
            min(N_BEST_START, len(start_values))
        )

        end_top = torch.topk(
            end_values,
            min(N_BEST_END, len(end_values))
        )

        candidates = []

        for start_position in start_top.indices:

            start_token = (
                context_start
                + start_position.item()
            )

            for end_position in end_top.indices:

                end_token = (
                    context_start
                    + end_position.item()
                )

                if end_token < start_token:
                    continue

                length = (
                    end_token
                    - start_token
                    + 1
                )

                if length > MAX_ANSWER_LENGTH:
                    continue

                start_char = offsets[start_token][0].item()
                end_char = offsets[end_token][1].item()

                if end_char <= start_char:
                    continue

                answer = context[
                    start_char:end_char
                ].strip()

                if not answer:
                    continue

                raw_score = (
                    start_logits[start_token].item()
                    + end_logits[end_token].item()
                )

                candidates.append({
                    "answer": answer,
                    "raw_score": raw_score,
                    "length": length
                })

        return candidates

    # --------------------------------------------------------
    # Select best candidate
    # --------------------------------------------------------

    def answer_from_context(
        self,
        question,
        context,
        retrieval_rank
    ):

        candidates = self.get_candidates(
            question,
            context
        )

        if not candidates:
            return "", 0.0

        best = None
        best_score = -float("inf")

        for candidate in candidates:

            answer = candidate["answer"]
            raw_score = candidate["raw_score"]

            # ----------------------------------------------
            # Question overlap
            # ----------------------------------------------

            overlap = question_overlap(
                question,
                answer
            )

            # ----------------------------------------------
            # Length handling
            #
            # Avoid strongly preferring tiny answers.
            # ----------------------------------------------

            length = candidate["length"]

            if length == 1:
                length_penalty = -0.8
            elif length == 2:
                length_penalty = -0.3
            else:
                length_penalty = 0.0

            # ----------------------------------------------
            # Retrieval rank
            #
            # Earlier retrieved passages receive a small
            # advantage, but the reader score dominates.
            # ----------------------------------------------

            retrieval_bonus = 0.15 / retrieval_rank

            # ----------------------------------------------
            # Final candidate score
            # ----------------------------------------------

            score = (
                raw_score
                + 1.5 * overlap
                + length_penalty
                + retrieval_bonus
            )

            if score > best_score:

                best_score = score
                best = candidate

        if best is None:
            return "", 0.0

        # Convert score into a rough confidence.
        confidence = 1 / (
            1 + math.exp(
                -max(
                    min(best_score, 10),
                    -10
                )
            )
        )

        return best["answer"], confidence

    # --------------------------------------------------------
    # Answer using retrieved passages
    # --------------------------------------------------------

    def answer(
        self,
        question,
        retrieved_hits
    ):

        best_answer = ""
        best_confidence = -1
        best_source = None

        for rank, hit in enumerate(
            retrieved_hits[:TOP_K],
            start=1
        ):

            answer, confidence = (
                self.answer_from_context(
                    question,
                    hit["text"],
                    rank
                )
            )

            if (
                answer
                and confidence > best_confidence
            ):

                best_answer = answer
                best_confidence = confidence

                best_source = {
                    "doc_id": hit["doc_id"],
                    "chunk_id": hit["chunk_id"]
                }

        return {
            "answer": best_answer,
            "confidence": best_confidence,
            "source": best_source
        }


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_questions():

    print("Loading QA dataset...")

    with open(
        QA_DATASET,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        if "test" in data:
            records = data["test"]

        elif "data" in data:
            records = data["data"]

        else:

            records = []

            for value in data.values():

                if isinstance(value, list):
                    records = value
                    break

    else:

        records = []

    print(
        f"Total records found: {len(records)}"
    )

    questions = []

    for record in records:

        question = (
            record.get("question")
            or record.get("query")
        )

        gold = (
            record.get("gold_answer")
            or record.get("answer")
            or record.get("gold")
        )

        if question and gold:

            questions.append({
                "question": question,
                "gold_answer": gold,
                "record": record
            })

    print(
        f"Questions with gold answers: "
        f"{len(questions)}"
    )

    return questions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMPROVED TRANSFORMER QA EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    retriever = DenseRetriever()

    retriever.build_index()

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    reader = TransformerQA()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    questions = load_test_questions()

    if not questions:

        print(
            "ERROR: No gold-answer questions found."
        )

        return

    results = []

    total_em = 0
    total_f1 = 0
    total_latency = 0

    print("\nRunning evaluation...\n")

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]
        gold = item["gold_answer"]

        # Retrieval
        hits = retriever.search(
            question,
            k=TOP_K
        )

        # Reader
        start_time = time.perf_counter()

        prediction = reader.answer(
            question,
            hits
        )

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        predicted = prediction["answer"]

        # Metrics
        em = exact_match(
            predicted,
            gold
        )

        f1 = f1_score(
            predicted,
            gold
        )

        total_em += em
        total_f1 += f1
        total_latency += latency

        result = {
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "exact_match": em,
            "f1": f1,
            "confidence": prediction[
                "confidence"
            ],
            "reader_latency_ms": latency,
            "source": prediction["source"]
        }

        results.append(result)

        print(
            f"[{i}/{len(questions)}] "
            f"EM={em} "
            f"F1={f1:.4f} "
            f"Latency={latency:.2f} ms"
        )

        print(
            f"Q: {question}"
        )

        print(
            f"Pred: {predicted}"
        )

        print(
            f"Gold: {gold}"
        )

        print("-" * 60)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    n = len(results)

    avg_em = total_em / n
    avg_f1 = total_f1 / n
    avg_latency = total_latency / n

    summary = {
        "model": QA_MODEL,
        "retrieval_model": (
            "BAAI/bge-small-en-v1.5"
        ),
        "top_k": TOP_K,
        "n_questions": n,
        "exact_match": avg_em,
        "f1": avg_f1,
        "reader_latency_ms_mean": avg_latency,
        "results": results
    }

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 60)
    print("IMPROVED TRANSFORMER QA RESULTS")
    print("=" * 60)

    print(
        f"Model:              {QA_MODEL}"
    )

    print(
        "Retrieval model:    "
        "BAAI/bge-small-en-v1.5"
    )

    print(
        f"Questions:          {n}"
    )

    print(
        f"Exact Match:        "
        f"{avg_em:.4f}"
    )

    print(
        f"F1:                 "
        f"{avg_f1:.4f}"
    )

    print(
        f"Mean QA latency:    "
        f"{avg_latency:.2f} ms"
    )

    print("=" * 60)

    print(
        f"\nResults saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()