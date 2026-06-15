# RAG Engineering-Document Assistant

A **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions
about engineering documents — FEM/design standards, analysis guidelines, and
software-tool documentation — and returns answers **grounded in the source text with
citations**.

> **Runnable demo:** [`rag_engineering_assistant.py`](rag_engineering_assistant.py)
> — runs out of the box with **no downloads and no API key**.

---

## Table of contents
1. [The problem RAG solves](#the-problem-rag-solves)
2. [How RAG works](#how-rag-works)
3. [What this project does](#what-this-project-does)
4. [Architecture](#architecture)
5. [Quick start (zero setup)](#quick-start-zero-setup)
6. [Optional upgrades](#optional-upgrades)
7. [Example questions](#example-questions)
8. [Using your own documents](#using-your-own-documents)
9. [Design choices explained](#design-choices-explained)
10. [Limitations & honesty notes](#limitations--honesty-notes)

---

## The problem RAG solves

Large language models are fluent but unreliable on specifics: ask one for "the acceptable
mesh aspect ratio in our standard" or "the fatigue safety factor for critical parts" and
it will **confidently invent** a plausible-sounding number that may be wrong. In
engineering, a confidently-wrong answer with no source is worse than no answer.

**RAG fixes this.** Instead of answering from memory, the system:
1. **Retrieves** the most relevant passages from *your* trusted documents, then
2. **Generates** an answer using *only* those passages — and **cites** them.

The result is traceable: every claim points back to a specific document and passage you
can verify. This is the architecture behind most production "chat with your documents"
and internal-knowledge-base tools.

---

## How RAG works

```
                                   ┌─────────────────────────────┐
   Your documents  ──ingest──▶     │  Chunk  →  Embed/Index      │   (done once)
   (.md/.txt/.pdf)                 └─────────────────────────────┘
                                                │
                                                ▼
   Question  ──────────────▶  ┌──────────────────────────────────┐
                              │  Retrieve top-k relevant chunks   │
                              └──────────────────────────────────┘
                                                │
                                                ▼
                              ┌──────────────────────────────────┐
                              │  Generate answer from chunks +    │
                              │  CITE sources   (or show passages)│
                              └──────────────────────────────────┘
                                                │
                                                ▼
                                   Grounded, cited answer
```

---

## What this project does

- **Ingests** a folder of engineering documents (Markdown, text, or PDF).
- **Chunks** each document into overlapping passages so retrieval is granular and no
  cross-boundary context is lost.
- **Indexes** the chunks for similarity search.
- **Retrieves** the most relevant passages for any question.
- **Answers** — either by generating a cited answer with an LLM (if configured) or, with
  no LLM, by returning the top passages with their sources and relevance scores.

It ships with four sample documents so it works immediately:
`fem_meshing_guidelines.md`, `fatigue_analysis_standard.md`,
`nonlinear_solver_settings.md`, `material_modeling_guide.md`.

---

## Architecture

| Stage | Default (zero-setup) | Optional upgrade |
|-------|----------------------|------------------|
| **Retrieval** | TF-IDF (scikit-learn) — lexical, instant, offline | Semantic embeddings (sentence-transformers) — understands meaning |
| **Generation** | Extractive — returns top passages + citations | LLM-generated cited answer (Anthropic or OpenAI) |
| **Documents** | `.md`, `.txt` | `.pdf` (via pypdf) |

This tiered design is deliberate: the project **runs for anyone instantly**, and each
component can be upgraded independently without changing the rest of the pipeline.

---

## Quick start (zero setup)

```bash
pip install numpy scikit-learn
python rag_engineering_assistant.py
```

That launches a short demo answer followed by an **interactive Q&A** prompt against the
sample documents. Or ask a single question directly:

```bash
python rag_engineering_assistant.py --query "How do I choose a fatigue safety factor?"
```

Example output (extractive mode, no LLM):

```
Q: What aspect ratio is acceptable for a structural mesh?
----------------------------------------------------------------
[1] fem_meshing_guidelines.md (relevance 0.508):
    ... Acceptable: aspect ratio below 5 for general structural regions.
    In regions of high stress gradient ... below 3. Aspect ratios above 10
    are not acceptable ...
Sources consulted: fem_meshing_guidelines.md
```

---

## Optional upgrades

### 1. Semantic retrieval (better recall)
TF-IDF matches keywords; semantic embeddings match *meaning* (e.g. it links "mesh
density" to "element size" even with no shared words).

```bash
pip install sentence-transformers
python rag_engineering_assistant.py --backend semantic
```

(First run downloads a small ~80 MB model; afterwards it is offline.)

### 2. LLM-generated cited answers
With an API key set, the system writes a natural-language answer synthesised from the
retrieved passages, with inline `[n]` citations:

```bash
pip install anthropic
set ANTHROPIC_API_KEY=sk-...        # Windows (use `export` on macOS/Linux)
python rag_engineering_assistant.py --query "What solver settings help nonlinear convergence?"
```

OpenAI is also supported (`pip install openai`, set `OPENAI_API_KEY`). If no key is set,
the system automatically falls back to extractive answers — it always works.

### 3. PDF documents
```bash
pip install pypdf
# drop .pdf files into sample_docs/ (or use --docs to point elsewhere)
```

---

## Example questions

The sample corpus can answer questions such as:
- "What aspect ratio is acceptable for a structural mesh?"
- "When should I use second-order elements?"
- "How do I choose a fatigue safety factor for a critical component?"
- "What is the difference between Goodman and Gerber mean-stress corrections?"
- "What contact settings help a nonlinear analysis converge?"
- "Which material model should I use for cyclic plasticity?"
- "How do I model a thermoplastic under sustained load?"

---

## Using your own documents

Point the assistant at any folder of documents:

```bash
python rag_engineering_assistant.py --docs "C:\path\to\my_standards" --k 5
```

Drop in your design standards, analysis procedures, tool manuals, or material datasheets
(as `.md`, `.txt`, or `.pdf`). No code changes needed — it ingests and indexes them on
startup. Use `--k` to control how many passages are retrieved per question.

> Note: keep proprietary/controlled documents local. This tool runs entirely on your
> machine in the default configuration (no data leaves your computer unless you enable an
> LLM API).

---

## Design choices explained

- **Overlapping chunks** — passages overlap by ~30 words so an answer spanning a chunk
  boundary is not split and lost.
- **TF-IDF default** — chosen so the project runs with dependencies most engineers already
  have, with no model download and full offline operation. Semantic search is a one-line
  upgrade when better recall is needed.
- **Extractive fallback** — even with no LLM, the system performs genuine RAG *retrieval*
  and returns cited passages, so it is always useful and always grounded.
- **Pluggable LLM** — generation is isolated behind one function that tries Anthropic, then
  OpenAI, then falls back. The retrieval pipeline is independent of the model.
- **Citations everywhere** — every answer reports which document(s) it came from, which is
  the entire point of RAG for engineering use.

---

## Limitations & honesty notes

- **Retrieval quality caps answer quality** — if the relevant passage is not retrieved, the
  answer cannot use it. Semantic retrieval and tuning `--k` and chunk size help.
- **No re-ranking or query rewriting** — production systems often add a re-ranker and
  multi-query expansion; this demo keeps the core pipeline clear instead.
- **Simple chunking** — paragraph-based with overlap. Very long tables or figures in PDFs
  may chunk imperfectly.
- **The LLM can still err** — even grounded, an LLM may misread context; the citations let a
  user verify, which is the safeguard.

This is a clean, correct, and genuinely useful RAG implementation that demonstrates the
full architecture — while being honest that production retrieval systems add further
layers for robustness.

**Stack:** Python · scikit-learn (TF-IDF) · optional: sentence-transformers, Anthropic/OpenAI SDK, pypdf
