*This project has been created as part of the 42 curriculum by sgil--de.*

# RAG Against the Machine

> A Python RAG (Retrieval-Augmented Generation) system that indexes the vLLM codebase and answers questions about it using BM25 retrieval + Ollama (Qwen) generation.

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
make install
```

### Ollama setup (required for generation)

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

### Index

```bash
uv run python -m student index --max_chunk_size 2000
```

### Tests

```bash
uv run pytest
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

### Evaluate

```bash
uv run python -m student evaluate
```

---

## Performance Analysis

```mermaid
graph TD
    Perf["Performance"]

    Index["Index ≤ 5 min"]
    Cold["Cold start ≤ 60s"]
    Warm["Warm ≤ 90s"]
    Recall["Recall ≥ target"]
    Answer["Answer ≤ 2s"]

    Perf --> Index
    Perf --> Cold
    Perf --> Warm
    Perf --> Recall
    Perf --> Answer
```

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
* [https://ollama.com/library/qwen2.5](https://ollama.com/library/qwen2.5)
* [https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)
* [https://docs.python.org/3/library/ast.html](https://docs.python.org/3/library/ast.html)
* [https://en.wikipedia.org/wiki/Retrieval-augmented_generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
