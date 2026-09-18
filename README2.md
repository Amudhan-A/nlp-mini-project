# NLP Mini Project --- Domain-Specific Question Answering

## Member 2 Handoff README

### Dense Retrieval + Transformer QA + Local LLM + Streamlit

This README documents the work completed for the **Dense Retrieval +
Transformer QA** part of the NLP mini project so that the next team
member can understand, run, evaluate, and integrate the system without
needing to reconstruct the implementation.

------------------------------------------------------------------------

## 1. Project Objective

The project implements a **domain-specific Question Answering (QA)
system** over the **NFCorpus (NutritionFacts Corpus)** dataset.

The selected domain is:

> **Medical & Nutrition Information Retrieval**

The system takes a natural-language question, retrieves the most
relevant document chunks using dense semantic retrieval, and then
generates an answer using a local Transformer-based language model.

### Final pipeline

``` text
User Question
      |
      v
BGE Dense Retrieval
(BAAI/bge-small-en-v1.5)
      |
      v
ChromaDB Vector Database
      |
      v
Top-5 Relevant Evidence
      |
      v
Llama 3.1 8B
(Local Ollama inference)
      |
      v
Grounded Natural-Language Answer
```

No paid API and no API key/token are required.

------------------------------------------------------------------------

# 2. Dataset Used

The project uses **NFCorpus**, obtained through the BEIR benchmark.

### Dataset details

-   Domain: Medical / Nutrition
-   Documents: approximately **3,633**
-   Generated chunks: **6,353**
-   Chunk size: **200 words**
-   Chunk overlap: **40 words**
-   Random seed used for dataset processing/evaluation: **42**

The processed chunks are stored in:

``` text
data/processed/chunks.json
```

The QA data is stored in:

``` text
data/qa_dataset.json
```

------------------------------------------------------------------------

# 3. Existing Project Structure

The important project files are:

``` text
nlp-mini-project/
│
├── data/
│   ├── raw/
│   │   └── nfcorpus/
│   │
│   ├── processed/
│   │   ├── chunks.json
│   │   └── queries_qrels.json
│   │
│   └── qa_dataset.json
│
├── results/
│   ├── baseline_results.json
│   ├── dense_retrieval_results.json
│   ├── transformer_qa_results.json
│   ├── llama_qa_results.json
│   ├── final_comparison.json
│   ├── error_analysis_10.json
│   └── error_analysis_10.md
│
├── prepare_data.py
├── build_qa_dataset.py
├── baseline.py
├── dense_retrieval.py
├── qa_reader.py
├── llama_qa.py
├── error_analysis.py
├── final_comparison.py
├── app.py
│
├── requirements.txt
└── requirements_updated.txt
```

------------------------------------------------------------------------

# 4. Work Completed by Member 2

The following components have been implemented and tested.

## 4.1 Dense Semantic Retrieval

Implemented using:

``` text
BAAI/bge-small-en-v1.5
```

Library:

``` text
sentence-transformers
```

The 6,353 document chunks are converted into dense vector embeddings.

The embeddings are stored in:

``` text
ChromaDB
```

Persistent database location:

``` text
data/vector_db/
```

Collection:

``` text
nfcorpus_dense
```

### Important implementation detail

ChromaDB has a batch-size limitation, so the 6,353 documents are
inserted in batches of 5,000 rather than in a single operation.

The system also checks whether the collection already contains
embeddings before rebuilding them.

Therefore, after the first successful indexing, running the retrieval
code does not unnecessarily regenerate all embeddings.

------------------------------------------------------------------------

# 5. Dense Retrieval Results

Evaluation was performed on **49 test questions**.

The final BGE dense retrieval results were:

  Metric             BGE Dense Retrieval
  ---------------- ---------------------
  Recall@1                        40.82%
  Recall@3                        61.22%
  Recall@5                        65.31%
  Recall@10                       67.35%
  MRR@10                          51.65%
  nDCG@10                         28.54%
  Mean latency                   15.5 ms
  Median latency                15.49 ms

The baseline BM25 results were:

  Metric              BM25
  -------------- ---------
  Recall@1          40.82%
  Recall@3          57.14%
  Recall@5          61.22%
  Recall@10         61.22%
  MRR@10            49.32%
  nDCG@10           26.22%
  Mean latency     4.08 ms

### Interpretation

Dense retrieval has the same Recall@1 as BM25 but improves retrieval at
larger cutoffs:

``` text
BM25 Recall@10      = 61.22%
BGE Recall@10       = 67.35%
```

It also improves:

``` text
MRR@10
nDCG@10
Recall@3
Recall@5
Recall@10
```

The trade-off is higher retrieval latency:

``` text
BM25 ≈ 4.08 ms
BGE  ≈ 15.5 ms
```

------------------------------------------------------------------------

# 6. Transformer QA --- DistilBERT

A Transformer extractive QA reader was also implemented using:

``` text
distilbert-base-cased-distilled-squad
```

The original attempt used the Hugging Face `pipeline()` API.

With the installed Transformers version, the `question-answering`
pipeline was not available through the expected task interface, so the
implementation was changed to the direct API:

``` text
AutoTokenizer
AutoModelForQuestionAnswering
```

This successfully loaded the model.

## Reader strategy

The reader receives the top retrieved passages and searches candidate
answer spans.

The improved implementation considers:

-   top candidate start/end positions
-   maximum answer length
-   question-term overlap
-   answer length penalty
-   retrieval rank bonus

This was used instead of simply taking the raw highest-scoring token
span.

## Results

On the 30 QA records containing gold answers:

``` text
Exact Match = 0.00%
F1          = 7.70%
```

Mean QA latency:

``` text
≈ 292 ms
```

The extractive reader was therefore retained as an experimental
Transformer baseline, but the project moved to a generative local model
for the final application.

------------------------------------------------------------------------

# 7. Final Transformer / Generative QA Model

The final QA generation component uses:

``` text
Llama 3.1 8B
```

through:

``` text
Ollama
```

Model:

``` text
llama3.1:8b
```

The model runs locally on the machine.

### No API key required

The system communicates with the local Ollama server:

``` text
http://localhost:11434/api/generate
```

Therefore:

-   No OpenAI API key
-   No Hugging Face token
-   No paid API
-   No cloud inference

is required.

------------------------------------------------------------------------

# 8. Llama QA Architecture

The Llama system works as follows:

``` text
Question
   |
   v
BGE Encoder
   |
   v
ChromaDB Similarity Search
   |
   v
Top 5 Chunks
   |
   v
Prompt Construction
   |
   v
Llama 3.1 8B
   |
   v
Answer
```

The prompt instructs Llama to:

1.  Use only the retrieved context.
2.  Not use outside knowledge.
3.  Not invent facts.
4.  Give a concise factual answer.
5.  State that there is insufficient information when the retrieved
    context does not contain enough evidence.

This makes the application a **retrieval-augmented QA system**.

------------------------------------------------------------------------

# 9. Llama QA Results

The final Llama evaluation was performed on **30 questions with gold
answers**.

  Metric             BGE + Llama 3.1
  ---------------- -----------------
  Exact Match                  0.00%
  F1                          15.26%
  Mean latency            4009.20 ms
  Median latency          3445.33 ms
  P95 latency             6214.89 ms

The Llama system produces substantially more natural and contextual
answers than the extractive DistilBERT reader, but generation is much
slower because the 8B model is running locally on CPU.

------------------------------------------------------------------------

# 10. Final System Comparison

The consolidated results are:

  --------------------------------------------------------------------------------------
  System          Recall@10       MRR@10      nDCG@10         EM           F1    Latency
  ------------ ------------ ------------ ------------ ---------- ------------ ----------
  BM25               61.22%       49.32%       26.22%      0.00%       15.11%    4.08 ms
  baseline                                                                    

  BGE dense      **67.35%**   **51.65%**   **28.54%**        ---          ---   15.50 ms
  retrieval                                                                   

  BGE +                 ---          ---          ---      0.00%        7.70%    ≈292 ms
  DistilBERT                                                                  

  BGE + Llama           ---          ---          ---      0.00%   **15.26%**   ≈4009 ms
  3.1                                                                         
  --------------------------------------------------------------------------------------

### Important comparison note

The BM25 F1 and Llama F1 come from different answer-generation stages.

BM25 uses the project's simpler sentence-overlap answer mechanism,
whereas Llama generates a natural-language answer from retrieved
evidence.

Therefore, the F1 values should not be described as a perfectly
apples-to-apples comparison of the retrieval algorithms.

The strongest direct retrieval comparison is:

``` text
BM25 Recall@10 = 61.22%
BGE Recall@10  = 67.35%
```

------------------------------------------------------------------------

# 11. Streamlit Application

A working Streamlit interface has been implemented in:

``` text
app.py
```

The interface provides:

-   Question input
-   Ask button
-   Generated answer
-   Top-5 retrieved evidence
-   Source/document IDs

Example interface flow:

``` text
Ask a question
      |
      v
"Does vitamin D reduce the risk of cancer?"
      |
      v
Ask
      |
      v
Answer
      |
      v
Retrieved Evidence
      |
      +-- Source 1
      +-- Source 2
      +-- Source 3
      +-- Source 4
      +-- Source 5
```

The application has been tested successfully in a browser through
Streamlit.

------------------------------------------------------------------------

# 12. Running the Application

## Start Ollama

Ollama should already be running in the background.

Check:

``` powershell
ollama list
```

The following model should be available:

``` text
llama3.1:8b
```

If Ollama is not running, start it with:

``` powershell
ollama serve
```

If the command reports:

``` text
Only one usage of each socket address
```

that means Ollama is already running.

------------------------------------------------------------------------

## Start Streamlit

From the project directory:

``` powershell
python -m streamlit run app.py --server.fileWatcherType none
```

Then open the local Streamlit page shown in the terminal, normally:

``` text
http://localhost:8501
```

The `--server.fileWatcherType none` option avoids irrelevant package
watcher warnings caused by the Transformers installation.

------------------------------------------------------------------------

# 13. Required Python Packages

The updated dependency list is:

``` text
pandas
scikit-learn
rank_bm25
nltk
beir
sentence-transformers
transformers
torch
chromadb
streamlit
```

The installed environment used during development included
approximately:

``` text
Python 3.14
PyTorch 2.14.0+cpu
Transformers 5.17.0
Sentence Transformers 6.1.0
Streamlit 1.64.0
```

These versions are included here mainly for reproducibility.

------------------------------------------------------------------------

# 14. Important Files for Member 3

### Dense retrieval

``` text
dense_retrieval.py
```

Responsible for:

-   loading NFCorpus chunks
-   loading BGE
-   generating embeddings
-   storing embeddings in ChromaDB
-   performing similarity search
-   calculating retrieval metrics
-   saving dense retrieval results

------------------------------------------------------------------------

### Extractive Transformer reader

``` text
qa_reader.py
```

Responsible for:

-   loading DistilBERT QA model
-   taking retrieved passages
-   extracting candidate answer spans
-   calculating EM/F1
-   saving Transformer QA results

------------------------------------------------------------------------

### Generative QA

``` text
llama_qa.py
```

Responsible for:

-   BGE retrieval
-   selecting top-5 evidence
-   creating the Llama prompt
-   calling local Ollama
-   generating answers
-   calculating EM/F1
-   measuring latency
-   saving results

------------------------------------------------------------------------

### Error analysis

``` text
error_analysis.py
```

Responsible for generating the 10-case error analysis.

Output:

``` text
results/error_analysis_10.json
results/error_analysis_10.md
```

------------------------------------------------------------------------

### Final comparison

``` text
final_comparison.py
```

Creates:

``` text
results/final_comparison.json
```

containing the consolidated system metrics.

------------------------------------------------------------------------

### Application

``` text
app.py
```

Responsible for the final Streamlit demonstration.

------------------------------------------------------------------------

# 15. Error Analysis

Ten low-performing cases were examined.

The main failure patterns are:

## 15.1 Retrieval mismatch

Some questions retrieve documents that are related to the question's
vocabulary but do not contain the exact information required.

Example:

``` text
Question: Chernobyl
```

The retrieved evidence included Fukushima studies and one
Chernobyl-related passage, but not enough information to reproduce the
gold answer.

------------------------------------------------------------------------

## 15.2 Short / ambiguous queries

Queries such as:

``` text
eggnog
chlorophyll
African-American
muscle health
```

contain very little explicit question context.

The retriever therefore has difficulty determining exactly what
information the user is requesting.

------------------------------------------------------------------------

## 15.3 Gold answer vs retrieved evidence mismatch

In several cases, the retrieved documents contain information related to
the topic but not the exact facts used in the gold answer.

For example, for:

``` text
muscle health
```

the retrieved documents discuss:

-   statin-associated muscle damage
-   muscle weakness
-   falls risk
-   lean tissue mass

but the gold answer focuses on:

-   essential amino acids
-   leucine
-   progressive resistance training

This indicates that the retrieval stage is an important source of
downstream QA error.

------------------------------------------------------------------------

## 15.4 Insufficient context

For:

``` text
Sometimes the Enzyme Myth Is True
```

the retrieved documents discuss other enzyme-related or dietary topics
but do not contain enough evidence to answer the specific question.

The model correctly avoided inventing an answer and returned an
insufficient-information response.

------------------------------------------------------------------------

## 15.5 Partial answers

For:

``` text
How Chemically Contaminated Are We?
```

the retrieved evidence contains useful information about:

-   mercury
-   arsenic
-   lead
-   dioxin
-   PBDEs
-   dietary exposure
-   environmental toxicants

However, it does not provide all the information needed for the gold
answer.

The generated answer therefore covers only part of the expected content.

------------------------------------------------------------------------

## 15.6 Over-answering

For some questions, Llama generates multiple related answers instead of
a concise response.

Example:

``` text
muscle health
```

The model generated several Q&A-style responses about statins and
alkaline diets.

This demonstrates a limitation of the prompt/model combination when the
query is very short or broad.

------------------------------------------------------------------------

# 16. Main Technical Limitations

The current system has several limitations that should be acknowledged
in the report.

### Retrieval limitations

-   Dense retrieval does not always retrieve the exact gold evidence.
-   Very short queries are ambiguous.
-   Semantically related documents can outrank the exact document.
-   Duplicate chunks/documents can appear in retrieved results.

### QA limitations

-   DistilBERT extractive QA performed poorly on the project's
    gold-answer format.
-   Generated answers may use different wording from the gold answer,
    resulting in low EM/F1 despite being semantically related.
-   Llama can produce verbose answers.
-   Llama may correctly refuse to answer when retrieval is insufficient,
    but this receives a low F1 score against a detailed gold answer.

### Performance limitations

The current environment runs Llama 3.1 8B locally on CPU.

Therefore:

``` text
Dense retrieval ≈ 15.5 ms
Llama generation ≈ 4.0 seconds average
```

The Llama generation stage is the primary latency bottleneck.

------------------------------------------------------------------------

# 17. Edge Cases Tested / Considerations

The project requirement includes handling:

### Empty input

The Streamlit application checks for an empty question and displays a
warning rather than executing retrieval.

### Out-of-vocabulary / unusual input

Dense embeddings allow semantically related queries to be handled even
when exact keywords are absent, although completely unrelated inputs may
still retrieve irrelevant documents.

### Very short queries

Queries such as:

``` text
eggnog
chlorophyll
Chernobyl
```

were observed to be challenging because they provide insufficient
intent.

### Very long input

Long questions may produce less precise retrieval because the embedding
represents a larger amount of text.

### Insufficient evidence

The Llama prompt explicitly instructs the model to respond:

``` text
Insufficient information in the retrieved context.
```

when the retrieved evidence does not support an answer.

### Latency

Latency is measured for the retrieval and QA stages.

------------------------------------------------------------------------

# 18. How to Explain Member 2's Work in the Presentation

A simple explanation is:

> "Our baseline uses BM25 keyword-based retrieval. I implemented dense
> semantic retrieval using BGE-small, where every document chunk is
> converted into a vector and stored in ChromaDB. For a question, we
> embed the question and retrieve the most semantically similar chunks.
> This improved Recall@10 from 61.22% with BM25 to 67.35% with BGE. I
> then connected the retrieved evidence to a Transformer-based QA
> component. We tested DistilBERT for extractive QA and Llama 3.1 8B for
> generative QA. The final Streamlit application uses BGE for retrieval
> and a locally running Llama model through Ollama to generate an answer
> grounded in the retrieved evidence."

------------------------------------------------------------------------

# 19. Architecture Explanation

If asked to explain the architecture:

``` text
                  +----------------+
                  |  User Question |
                  +-------+--------+
                          |
                          v
                  +----------------+
                  | BGE Encoder    |
                  +-------+--------+
                          |
                          v
                  +----------------+
                  |   ChromaDB     |
                  | Vector Search  |
                  +-------+--------+
                          |
                    Top-5 Chunks
                          |
                          v
                  +----------------+
                  | Prompt Builder |
                  +-------+--------+
                          |
                          v
                  +----------------+
                  |  Llama 3.1 8B  |
                  |    Ollama      |
                  +-------+--------+
                          |
                          v
                  +----------------+
                  | Final Answer   |
                  +----------------+
```

------------------------------------------------------------------------

# 20. What Member 3 Needs to Do Next

The Member 2 implementation should be treated as complete unless
integration reveals a bug.

Member 3 can now focus on:

1.  Integrating the evaluation results.
2.  Incorporating the 10-case error analysis into the final report.
3.  Combining the baseline, dense retrieval, and QA metrics.
4.  Preparing the final evaluation section.
5.  Preparing the Streamlit demo portion.
6.  Preparing the presentation.
7.  Running the complete project once from a clean start before
    submission.

### Do not unnecessarily change

Unless required for integration, do not replace:

``` text
BAAI/bge-small-en-v1.5
```

or:

``` text
llama3.1:8b
```

The current system is already working end-to-end.

------------------------------------------------------------------------

# 21. Quick Start for Member 3

From the repository root:

### Check Python

``` powershell
python --version
```

### Check Ollama

``` powershell
ollama list
```

Make sure:

``` text
llama3.1:8b
```

is present.

### Run the Streamlit application

``` powershell
python -m streamlit run app.py --server.fileWatcherType none
```

### Test question

``` text
Does vitamin D reduce the risk of cancer?
```

The application should return an answer and display the retrieved
evidence.

------------------------------------------------------------------------

# 22. Important Results Files

Use these files when preparing the final report:

``` text
results/baseline_results.json
```

Classical BM25/TF-IDF results.

``` text
results/dense_retrieval_results.json
```

BGE dense retrieval results.

``` text
results/transformer_qa_results.json
```

DistilBERT extractive QA results.

``` text
results/llama_qa_results.json
```

Llama generative QA results.

``` text
results/final_comparison.json
```

Consolidated comparison.

``` text
results/error_analysis_10.md
```

Readable error-analysis results.

``` text
results/error_analysis_10.json
```

Machine-readable error-analysis results.

------------------------------------------------------------------------

# 23. Final Status

## Member 2 --- Dense Retrieval + Transformer QA

**STATUS: COMPLETE**

Implemented:

-   [x] NFCorpus chunk embedding
-   [x] BGE dense retrieval
-   [x] ChromaDB vector storage
-   [x] Retrieval evaluation
-   [x] DistilBERT extractive Transformer QA
-   [x] Llama 3.1 8B generative QA
-   [x] Local Ollama inference
-   [x] No API key/token dependency
-   [x] QA EM/F1 evaluation
-   [x] Latency measurement
-   [x] 10-case error analysis
-   [x] Final comparison
-   [x] Streamlit application
-   [x] End-to-end testing

The current implementation is ready for integration with the rest of the
project.
