"""
RAG Engineering-Document Assistant
==================================
A Retrieval-Augmented Generation (RAG) system that answers natural-language questions
about a corpus of engineering documents — FEM/design standards, analysis guidelines,
and software-tool documentation — and returns answers grounded in the source text WITH
CITATIONS.

Why RAG? A plain language model will confidently invent ("hallucinate") engineering
numbers and clauses. RAG fixes this by first RETRIEVING the most relevant passages from
YOUR trusted documents, then asking the model to answer using ONLY those passages — and
to cite them. The answer is traceable to a source, which is essential in engineering.

DESIGN GOAL: runs out of the box with NO downloads and NO API key.
  - Retrieval backend (default): TF-IDF (scikit-learn) — instant, offline.
  - Optional upgrade: semantic embeddings (sentence-transformers) for better recall.
  - Generation (default): extractive — returns the top passages with citations.
  - Optional upgrade: LLM-generated answers (Anthropic or OpenAI) if an API key is set.

Usage:
    python rag_engineering_assistant.py                      # interactive Q&A
    python rag_engineering_assistant.py --query "your question"
    python rag_engineering_assistant.py --backend semantic   # if sentence-transformers installed
    python rag_engineering_assistant.py --docs ./sample_docs --k 4

Author: Mohd Aaquib Khan
"""

import os
import re
import glob
import argparse
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_documents(docs_dir):
    """
    Load all .md / .txt files (and .pdf if pypdf is installed) from a directory.
    Returns a list of (filename, full_text).
    """
    docs = []
    paths = sorted(glob.glob(os.path.join(docs_dir, "*")))
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        if ext in (".md", ".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((name, f.read()))
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
                docs.append((name, text))
            except ImportError:
                print(f"  [skip] {name}: install 'pypdf' to read PDF files")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text, max_words=120, overlap=30):
    """
    Split a document into overlapping chunks of ~max_words words.
    Overlap preserves context across chunk boundaries so an answer that straddles
    two chunks isn't lost. Splits on paragraph boundaries where possible.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current, count = [], [], 0
    for para in paragraphs:
        words = para.split()
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            # start next chunk with an overlap tail of the previous one
            tail = " ".join(current).split()[-overlap:]
            current, count = list(tail), len(tail)
        current.extend(words)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunk_index(docs, max_words=120, overlap=30):
    """Flatten all documents into a list of chunk records."""
    records = []
    for fname, text in docs:
        for i, chunk in enumerate(chunk_text(text, max_words, overlap)):
            records.append({"source": fname, "chunk_id": i, "text": chunk})
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETRIEVAL BACKENDS
# ─────────────────────────────────────────────────────────────────────────────
class TfidfRetriever:
    """Lexical retrieval — fast, offline, no model download. Good baseline."""
    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def query(self, text, k=4):
        from sklearn.metrics.pairwise import linear_kernel
        q = self.vectorizer.transform([text])
        scores = linear_kernel(q, self.matrix).flatten()
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


class SemanticRetriever:
    """
    Semantic retrieval using sentence-transformer embeddings — understands meaning,
    not just keyword overlap (e.g. matches "mesh density" to "element size").
    Requires: pip install sentence-transformers
    """
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True)

    def query(self, text, k=4):
        q = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.embeddings @ q                 # cosine (vectors normalised)
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def make_retriever(backend, chunks):
    if backend == "semantic":
        try:
            return SemanticRetriever(chunks)
        except ImportError:
            print("  [warn] sentence-transformers not installed — falling back to TF-IDF.")
            print("         (pip install sentence-transformers for semantic search)")
    return TfidfRetriever(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def build_context(results):
    """Format retrieved chunks into a numbered, citable context block."""
    blocks = []
    for n, (chunk, score) in enumerate(results, 1):
        blocks.append(f"[{n}] (source: {chunk['source']}, chunk {chunk['chunk_id']})\n{chunk['text']}")
    return "\n\n".join(blocks)


def generate_with_llm(query, context):
    """
    Generate a grounded answer with an LLM if an API key is available.
    Tries Anthropic first, then OpenAI. Returns None if neither is configured.
    """
    prompt = (
        "You are an engineering documentation assistant. Answer the question using ONLY "
        "the context below. Cite sources with their [number]. If the answer is not in the "
        "context, say so plainly.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            print(f"  [warn] Anthropic call failed: {e}")

    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  [warn] OpenAI call failed: {e}")

    return None


def extractive_answer(results):
    """
    Fallback 'answer' with no LLM: present the most relevant passages with citations.
    This is genuine RAG retrieval — just without the generation step.
    """
    if not results:
        return "No relevant passage found in the documents."
    lines = ["(No LLM configured — showing the most relevant source passages.)\n"]
    for n, (chunk, score) in enumerate(results, 1):
        snippet = chunk["text"]
        if len(snippet) > 400:
            snippet = snippet[:400] + " ..."
        lines.append(f"[{n}] {chunk['source']} (relevance {score:.3f}):\n    {snippet}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RAG ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
class RAGAssistant:
    def __init__(self, docs_dir, backend="tfidf", max_words=120, overlap=30):
        docs = load_documents(docs_dir)
        if not docs:
            raise SystemExit(f"No documents found in {docs_dir}")
        self.chunks = build_chunk_index(docs, max_words, overlap)
        print(f"  Ingested {len(docs)} documents -> {len(self.chunks)} chunks "
              f"(backend: {backend})")
        self.retriever = make_retriever(backend, self.chunks)

    def ask(self, query, k=4):
        results = self.retriever.query(query, k=k)
        context = build_context(results)
        answer = generate_with_llm(query, context)
        if answer is None:
            answer = extractive_answer(results)
        sources = sorted({c["source"] for c, _ in results})
        return answer, sources, results


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="RAG assistant for engineering documents")
    ap.add_argument("--docs", default=os.path.join(here, "sample_docs"),
                    help="folder of .md/.txt/.pdf documents")
    ap.add_argument("--backend", choices=["tfidf", "semantic"], default="tfidf")
    ap.add_argument("--k", type=int, default=4, help="number of passages to retrieve")
    ap.add_argument("--query", default=None, help="ask one question and exit")
    args = ap.parse_args()

    print("=" * 64)
    print("  RAG ENGINEERING-DOCUMENT ASSISTANT")
    print("=" * 64)
    rag = RAGAssistant(args.docs, backend=args.backend)

    def handle(q):
        answer, sources, _ = rag.ask(q, k=args.k)
        print("\n" + "-" * 64)
        print(f"Q: {q}")
        print("-" * 64)
        print(answer)
        print(f"\nSources consulted: {', '.join(sources)}")

    if args.query:
        handle(args.query)
        return

    # Demo questions, then interactive
    print("\n  Example questions (try your own after):")
    demo = [
        "What aspect ratio is acceptable for a structural mesh?",
        "How do I choose a fatigue safety factor?",
        "What solver settings help a nonlinear analysis converge?",
    ]
    for q in demo:
        print(f"    - {q}")
    print()
    handle(demo[0])

    print("\n" + "=" * 64)
    print("  Interactive mode — type a question, or 'quit' to exit.")
    print("=" * 64)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q", ""):
            break
        handle(q)


if __name__ == "__main__":
    main()
