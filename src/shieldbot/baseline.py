"""Step 2 - Baseline RAG (deliberately naive).

This is the "before" system for the evaluation. It makes the classic mistakes on purpose:
  - fixed 500-character chunks that ignore headings and cut tables in half
  - vector search only (no BM25)
  - no metadata filters, so superseded and internal docs can be retrieved
  - no reranking, no query rewriting, no guardrails
Every later step fixes one of these, and the evaluation shows how much each fix helps.
"""
import numpy as np

from .models import chat, embed_passages, embed_query


def naive_chunks(documents: list[dict], size: int = 500, overlap: int = 50) -> list[dict]:
    """Cut each document body into fixed-size character windows."""
    chunks = []
    step = size - overlap
    for doc in documents:
        text = doc["text"]
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({
                    "id": f"{doc['metadata']['doc_id']}::naive::{i}",
                    "text": piece,
                    "metadata": doc["metadata"],
                })
    return chunks


class BaselineRAG:
    """Naive RAG: embed chunks once, then cosine search + a simple answer prompt."""

    name = "baseline"

    def __init__(self, documents: list[dict]):
        self.chunks = naive_chunks(documents)
        self.embeddings = embed_passages([c["text"] for c in self.chunks])
        print(f"Baseline index: {len(self.chunks)} chunks from {len(documents)} docs")

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        """Return the top-k chunks by cosine similarity (vectors are normalised)."""
        scores = self.embeddings @ embed_query(question)
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def answer(self, question: str, k: int = 5) -> dict:
        chunks = self.retrieve(question, k)
        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        messages = [
            {"role": "system", "content": "You are an insurance assistant. Answer the question using the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return {"answer": chat(messages), "chunks": chunks}