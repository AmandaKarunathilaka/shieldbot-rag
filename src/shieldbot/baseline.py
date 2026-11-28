"""The baseline: a deliberately basic RAG, the "before" system for the evaluation (R10).

It uses the SAME embedding model and the SAME LLM as the final ShieldBot, so the
comparison measures only our improvements. It makes the classic mistakes on purpose:
  - fixed 500-character chunks that ignore headings and cut tables in half
  - no context header, so a chunk does not say which product it is about
  - vector search only (no BM25), no reranker, no relevance check
  - no metadata filters, so superseded and internal docs can be retrieved
  - no query rewriting, no guardrails, a one-line prompt
"""
import uuid

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import config


def naive_chunks(documents: list[Document], size: int = 500, overlap: int = 50) -> list[Document]:
    """Cut each document body into fixed-size character windows."""
    chunks = []
    step = size - overlap
    for doc in documents:
        text = doc.page_content
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append(Document(
                    page_content=piece,
                    metadata={**doc.metadata, "section": "(500-char window)",
                              "chunk_id": f"{doc.metadata['doc_id']}::naive::{i}"},
                ))
    return chunks


def to_dicts(chunks, scores=None) -> list[dict]:
    """The shape evaluation.py expects: {'text', 'metadata', 'score'}."""
    scores = scores or [None] * len(chunks)
    return [{"text": c.page_content, "metadata": c.metadata, "score": s} for c, s in zip(chunks, scores)]


class BaselineRAG:
    """Naive RAG: an in-memory Chroma index, top-k vector search and a simple prompt."""

    name = "baseline"

    def __init__(self, documents: list[Document], embedding_model: str = config.EMBEDDING_MODEL, embeddings=None):
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        self.chunks = naive_chunks(documents)
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings or OpenAIEmbeddings(model=embedding_model, max_retries=5),
            ids=[c.metadata["chunk_id"] for c in self.chunks],
            collection_name=f"baseline_{uuid.uuid4().hex[:8]}",   # in memory, a fresh one each run
        )
        print(f"Baseline index: {len(self.chunks)} chunks from {len(documents)} docs (in memory)")

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        return to_dicts(self.vectorstore.similarity_search(question, k=k))

    def answer(self, question: str, k: int = 5) -> dict:
        from .chatbot import get_llm   # the same LLM as the final system

        chunks = self.retrieve(question, k)
        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        messages = [
            SystemMessage(content="You are an insurance assistant. Answer the question using the context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        return {"answer": get_llm().invoke(messages).content, "chunks": chunks}