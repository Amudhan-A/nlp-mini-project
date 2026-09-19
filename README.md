# Question Answering over a Domain-Specific Document Collection Using Dense Retrieval

## Member 1 Deliverables: Dataset + Baseline Retrieval

### 1. Domain Document Collection & Metadata
- **Domain**: Medical & Nutrition Information Retrieval
- **Dataset**: NFCorpus (NutritionFacts Corpus, part of the BEIR benchmark)
- **Source**: UKP Lab, Technische Universität Darmstadt ([Download Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip))
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Document Count**: 3,633 medical abstracts / full articles
- **Chunk Count**: 6,353 passages
- **Chunking Strategy**: 200 words per chunk with 40-word sliding overlap (`CHUNK_SIZE_WORDS = 200`, `CHUNK_OVERLAP_WORDS = 40`)
- **Processed Corpus Size**: ~7.0 MB (`data/processed/chunks.json`)

---

### 2. Evaluation Dataset (`data/qa_dataset.json`)
- **Total Records**: 333 queries
  - **Train**: 226 queries (~68%)
  - **Validation**: 48 queries (~14%)
  - **Test**: 59 queries (~18%)
  - **Edge Cases**: 10 hand-written adversarial queries (empty, whitespace, out-of-domain, code-mixed, OOV, ambiguous, very long, unseen information, non-linguistic)
- **Reproducibility**: Fixed random seed (`seed = 42`)
- **Ground-Truth QA Annotations**: 30 test-split questions annotated with factual ground-truth answer strings derived directly from relevant PubMed abstracts.

---

### 3. Classical Baseline Results Summary

Evaluated on the 49 test-split queries using `python baseline.py --compare`:

| Metric | TF-IDF Baseline | BM25 Baseline | Winner |
| :--- | :--- | :--- | :--- |
| **Recall@1** | 38.78% | **40.82%** | **BM25** |
| **Recall@3** | 55.10% | **57.14%** | **BM25** |
| **Recall@5** | 57.14% | **61.22%** | **BM25** |
| **Recall@10** | 57.14% | **61.22%** | **BM25** |
| **MRR@10** | 0.4677 | **0.4932** | **BM25** |
| **nDCG@10** | 0.2343 | **0.2622** | **BM25** |
| **Exact Match (EM)** | 0.00% | 0.00% | Tie |
| **Token F1** | 14.14% | **15.11%** | **BM25** |
| **Latency (Mean)** | **2.14 ms** | 4.08 ms | **TF-IDF** |
| **Index Build Time** | 13.86 s | **1.25 s** | **BM25** |
| **Peak Index RAM** | 169.73 MB | **70.20 MB** | **BM25** |

Full per-query results, latencies, and edge-case logs are stored in `results/baseline_results.json`.

---

### 4. Team Handoff Instructions

#### For Member 2 (Dense Retrieval + Transformer QA):
- **Corpus to embed**: `data/processed/chunks.json` (contains `chunk_id`, `doc_id`, `text`).
- **Evaluation queries & relevance**: `data/qa_dataset.json`.
- **Target to beat**:
  - Retrieval: BM25 **Recall@5 = 61.22%**, **MRR@10 = 0.4932**.
  - Reader QA: Naive extractive sentence F1 = **15.11%**.

#### For Member 3 (Evaluation + Application + Integration):
- **Import baseline in app**:
  ```python
  from baseline import BaselineRetriever
  retriever = BaselineRetriever.from_files(method="bm25")
  result = retriever.answer("Is coconut milk good for you?")
  # returns: {"answer": str, "confidence": float, "sources": list, "status": str}
  ```
- **Error analysis data**: Check `results/baseline_results.json` (`per_question` array and `edge_cases` array) for pre-computed failure cases.



---
## Member 2 Deliverables: Dense Retrieval + Transformer QA

### Dense Retrieval
- **Model**: `BAAI/bge-small-en-v1.5`, cosine similarity, ChromaDB vector store at `data/vector_db/` (gitignored — rebuild locally with `python dense_retrieval.py`)

### Transformer Reader QA
- **Model**: `distilbert-base-cased-distilled-squad`, extractive span answering over top-5 dense-retrieved chunks
- **Run**: `python qa_reader.py`

### Generative QA (Llama 3.1 via Ollama)
- **Model**: `llama3.1:8b` via local Ollama
- **Run**: `python llama_qa.py`

---

## Member 3 Deliverables: Evaluation + Application Integration

### Unified Evaluation Pipeline
- **Streamlit demo**: `streamlit run app.py` (needs Ollama running)
- **Error analysis**: `python error_analysis.py` → `results/error_analysis_10.md`
- **Edge-case robustness**: `python edge_case_eval.py` → `results/dense_edge_cases.json`
- **Unified comparison**: `python final_comparison.py` → `results/final_comparison.json`
- **Full pipeline orchestrator**: `python run_evaluation.py` (`--skip-llama` if Ollama isn't running) — runs every stage above in dependency order

### Results Summary (49-query test split, 30 gold-answer QA subset)

| Metric | BM25 | BGE Dense | BGE + DistilBERT | BGE + Llama 3.1 |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1** | 40.82% | 42.86% | – | – |
| **Recall@5** | 61.22% | 65.31% | – | – |
| **Recall@10** | 61.22% | 67.35% | – | – |
| **MRR@10** | 0.4932 | 0.5264 | – | – |
| **nDCG@10** | 0.2622 | 0.2871 | – | – |
| **Exact Match** | 0.00% | – | 0.00% | 0.00% |
| **Token F1** | 15.11% | – | 7.70% | 15.26% |

Full breakdown (including Recall@1/3/5 for every system) is in `results/final_comparison.json`.

> **Note on Exact Match**: EM is 0.00% for every QA system by construction — gold answers are full abstractive sentences, not extractable spans, so no system can match them verbatim. **Token F1** is the meaningful metric. Dense retrieval improves ranking over BM25 across every retrieval metric; DistilBERT's F1 is dragged down by the many single-word/title-style questions (e.g. `"eggnog"`, `"salmon"`) that lack the interrogative structure a SQuAD-trained extractive reader needs to localize a span.

### Edge-Case Robustness Analysis (10 hand-written adversarial queries)

Dense retrieval + DistilBERT were run against the same 10 edge cases as the BM25 baseline, for direct comparison (`results/dense_edge_cases.json` vs the `edge_cases` array in `results/baseline_results.json`). Two gaps found relative to the classical baseline, which explicitly detects and rejects these inputs:
- **Empty/whitespace input is not rejected** by the dense pipeline — it returns a top-5-chunk answer with no guard, whereas BM25 correctly returns *"I don't have a question to process."*
- **DistilBERT crashes on unusually long input** with a tokenizer truncation error (1 of 10 edge cases), where BM25 truncates gracefully and still returns an answer.

### Reproducing These Results — Required Run Order

`data/vector_db/` is not committed. Run in order:

```bash
pip install -r requirements.txt
python prepare_data.py
python build_qa_dataset.py
python baseline.py --compare
python dense_retrieval.py
python qa_reader.py
python llama_qa.py
python error_analysis.py
python edge_case_eval.py
python final_comparison.py
```

Or run everything at once: `python run_evaluation.py` (`--skip-llama` if Ollama isn't running).