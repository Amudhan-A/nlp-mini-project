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
- **Total Records**: 335 queries
  - **Train**: 227 queries (70%)
  - **Validation**: 49 queries (15%)
  - **Test**: 49 queries (15%)
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
