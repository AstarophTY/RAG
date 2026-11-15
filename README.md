*This project has been created as part of the 42 curriculum by sgil--de.*

# RAG Against the Machine

> A Python RAG (Retrieval-Augmented Generation) system that indexes the vLLM codebase and answers questions about it using BM25 retrieval + Ollama (Qwen) generation.

**Results on the public datasets (100 questions each):** recall@5 = **0.74** on documentation questions and
**0.66** on code questions; indexing of the 1,826 files takes about 10 s and retrieval runs at hundreds of
questions per second. See [Performance Analysis](#performance-analysis).

---

## Table of Contents

1. [Description](#description)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Chunking Strategy](#chunking-strategy)
5. [Retrieval Method](#retrieval-method)
6. [Data Models](#data-models)
7. [Instructions](#instructions)
8. [Pipeline Usage (CLI)](#pipeline-usage-cli)
9. [Performance Analysis](#performance-analysis)
10. [Design Decisions](#design-decisions)
11. [Challenges Faced](#challenges-faced)
12. [Resources](#resources)

---

## Description

RAG Against the Machine is a Retrieval-Augmented Generation system built to answer questions about the vLLM codebase.

It:
- parses a repository,
- chunks files,
- indexes them using BM25,
- retrieves relevant chunks,
- generates answers using a local LLM.

### Goals

- Ingestion: parse and chunk files  
- Retrieval: find relevant chunks  
- Generation: produce grounded answers  
- Evaluation: measure recall@k  

---

## System Architecture

### Global pipeline

```mermaid
graph TD
    CLI["CLI"]

    Ingestion["Ingestion"]
    Walk["Walk repo"]
    Filter["Filter files"]
    Chunk["Chunk files"]

    PyChunk["Python AST"]
    TxtChunk["Text sliding window"]

    Save["Save chunks"]

    Index["Index"]
    CodeIndex["BM25 code"]
    DocsIndex["BM25 docs"]

    Retrieval["Retrieval"]
    Query["Query"]
    Tokenize["Tokenize"]
    ScoreCode["Score code"]
    ScoreDocs["Score docs"]
    Merge["Merge"]
    TopK["Top-k"]

    Generation["Generation"]
    Context["Context"]
    Model["Ollama (Qwen)"]
    Answer["Answer"]

    Eval["Evaluation"]
    Recall["Recall@k"]

    CLI --> Ingestion
    Ingestion --> Walk --> Filter --> Chunk
    Chunk --> PyChunk --> Save
    Chunk --> TxtChunk --> Save

    Save --> Index
    Index --> CodeIndex
    Index --> DocsIndex

    CLI --> Retrieval
    Retrieval --> Query --> Tokenize
    Tokenize --> ScoreCode
    Tokenize --> ScoreDocs
    ScoreCode --> Merge
    ScoreDocs --> Merge
    Merge --> TopK

    TopK --> Generation
    Generation --> Context --> Model --> Answer

    Answer --> Eval --> Recall
```

---

### Data flow for a single query

```mermaid
graph TD
    User["User Question"]

    Retriever["Retriever"]
    CodeIndex["BM25 Code Index"]
    DocsIndex["BM25 Docs Index"]

    Merge["Merge + Rank"]
    TopK["Top-k Sources"]

    Generator["Generator"]
    Context["Context window"]
    LLM["Ollama: qwen2.5:1.5b"]

    Output["Final Answer"]

    User --> Retriever
    Retriever --> CodeIndex
    Retriever --> DocsIndex

    CodeIndex --> Merge
    DocsIndex --> Merge

    Merge --> TopK

    TopK --> Generator
    Generator --> Context --> LLM --> Output
```

---

## Project Structure

```txt
rag-against-the-machine/
│
├── src/
│   ├── __main__.py               ← CLI entry point (Python Fire)
│   ├── models.py                 ← All Pydantic data models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── walker.py             ← Walk repo, filter by extension
│   │   ├── chunker_python.py     ← AST-based Python chunker
│   │   ├── chunker_text.py       ← Sliding-window text/Markdown chunker
│   │   └── chunk_store.py        ← Save / load chunks JSON to disk
│   │
│   ├── index/
│   │   ├── __init__.py
│   │   └── bm25_index.py         ← Build, save, load BM25 indexes
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          ← Query → top-k MinimalSource
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py          ← Context assembly + Ollama inference
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py          ← recall@k metric
│
├── data/                         ← NOT committed (.gitignore)
│   ├── raw/
│   │   └── vllm-0.10.1/          ← vLLM source repository
│   ├── processed/
│   │   ├── chunks/
│   │   │   ├── code_chunks.json
│   │   │   └── docs_chunks.json
│   │   └── bm25_index/
│   │       ├── code/
│   │       └── docs/
│   ├── datasets/
│   │   ├── AnsweredQuestions/
│   │   │   ├── dataset_code_public.json
│   │   │   └── dataset_docs_public.json
│   │   └── UnansweredQuestions/
│   │       ├── dataset_code_public.json
│   │       └── dataset_docs_public.json
│   └── output/
│       ├── search_results/
│       └── search_results_and_answer/
│
├── tests/
│   ├── test_chunker_python.py
│   ├── test_chunker_text.py
│   ├── test_retriever.py
│   └── test_evaluator.py
│
├── pyproject.toml                ← uv project config + dependencies
├── uv.lock                       ← Locked dependency tree
├── Makefile
├── .gitignore
└── README.md
```

---

## Chunking Strategy

### Python (AST)

```mermaid
graph TD
    File["Python File"]
    AST["Parse AST"]

    Class["Class"]
    Function["Function"]

    Chunk1["Chunk"]
    Chunk2["Chunk"]

    File --> AST
    AST --> Class --> Chunk1
    AST --> Function --> Chunk2
```

---

### Text (Sliding Window)

```mermaid
graph TD
    Doc["Document"]

    Sections["Sections"]
    Paragraphs["Paragraphs"]

    Long["Long text"]
    Window1["Window 1"]
    Window2["Window 2 (overlap)"]

    Doc --> Sections --> Paragraphs
    Paragraphs --> Long
    Long --> Window1
    Long --> Window2
```

---

## Retrieval Method

### BM25 Flow

```mermaid
graph TD
    Query["Query"]

    Code["BM25 Code Index"]
    Docs["BM25 Docs Index"]

    ScoreC["Scores (code)"]
    ScoreD["Scores (docs)"]

    Merge["Merge"]
    TopK["Top-k"]

    Query --> Code --> ScoreC
    Query --> Docs --> ScoreD

    ScoreC --> Merge
    ScoreD --> Merge

    Merge --> TopK
```

---

## Data Models

```mermaid
graph TD
    Source["Source"]

    Question["Question"]
    Answered["Answered Question"]

    Dataset["Dataset"]

    Search["Search Result"]
    SearchAnswer["Search + Answer"]

    StudentSearch["Student Search"]
    StudentFinal["Final Result"]

    Question -->|can become| Answered
    Answered -->|uses| Source

    Dataset -->|contains| Question
    Dataset -->|contains| Answered

    Question --> Search
    Search --> Source
    Search --> SearchAnswer

    StudentSearch --> Search
    StudentFinal --> SearchAnswer
```

---

## Instructions

### Installation

```bash
make install   # uv sync + ollama pull qwen3:0.6b
make data      # extract the vLLM sources from data/raw/vllm-0.10.1.zip
```

### Ollama setup (required for generation only)

```bash
ollama serve
```

The default model is `qwen3:0.6b`; another one can be passed with `--model_name`.

### Index

```bash
make index
# or: uv run python -m student index --max_chunk_size 2000
```

### Tests and lint

```bash
make test   # pytest: chunkers and evaluator
make lint   # flake8 + mypy
```

---

## Pipeline Usage (CLI)

### Search

```bash
uv run python -m student search "question" --k 5
```

### Answer

```bash
uv run python -m student answer "question" --k 10
```

### Evaluate on a dataset

```bash
# 1. Retrieve the top 10 sources for every question
uv run python -m student search_dataset data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10

# 2. Compute recall@1/3/5/10 against the ground truth
uv run python -m student evaluate \
  data/output/search_results/dataset_docs_public_k10_search_results.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
```

---

## Performance Analysis

Measured with `search_dataset` + `evaluate` (k = 10) on the public datasets. A retrieved chunk counts as a match
when it is in the right file and overlaps at least 5% of the ground-truth span.

| Dataset (100 questions) | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Documentation questions | 0.51 | 0.69 | 0.74 | 0.81 |
| Code questions | 0.51 | 0.61 | 0.66 | 0.73 |

| Step | Time |
| --- | --- |
| Indexing (1,826 files → 28,305 code chunks + 6,162 docs chunks) | ~10 s |
| Retrieval for 100 questions | < 1 s |

Code questions are harder for BM25: identifiers rarely appear in the natural-language question, which is the
main lead for improvement (identifier splitting, or a hybrid BM25 + embeddings retriever).

---

## Design Decisions

* BM25 for fast retrieval
* AST chunking for code accuracy
* Separate indexes for code and docs
* Pydantic for validation
* Singleton model loading

---

## Challenges Faced

* Token limit → context truncation
* Model latency → caching
* Code recall → smaller chunks
* Encoding issues → ignore errors

---

## Resources

* [https://en.wikipedia.org/wiki/Okapi_BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
* [https://ollama.com/library/qwen3](https://ollama.com/library/qwen3)
* [https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)
* [https://docs.python.org/3/library/ast.html](https://docs.python.org/3/library/ast.html)
* [https://en.wikipedia.org/wiki/Retrieval-augmented_generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
