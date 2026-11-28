"""Notebook 2, Sections 3-9 - finding the right chunks.

    Open the database (manifest) -> Mode filter (R3, R4) -> BM25 + vector search
    -> Hybrid (RRF, R5) -> Reranker (R6) -> Relevance check -> Cross-references

The filter is applied INSIDE every search, so a superseded or internal chunk
never reaches the reranker or the LLM. The AI cannot leak text it never received.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import config

# =====================================================================
# Section 4 - Who can see what (version + audience control)
# =====================================================================

# Customer mode: only active, public documents (R3 + R4).
# Staff mode: active documents, including internal ones (Bonus B1).
# Superseded documents are blocked in BOTH modes (R3).
MODE_CONDITIONS = {
    "customer": [{"status": "active"}, {"audience": "public"}],
    "staff": [{"status": "active"}],
}


def normalise_mode(mode) -> str:
    """'Customer', 'STAFF', None -> 'customer' / 'staff'. Unknown modes get the lowest access."""
    mode = str(mode or "customer").strip().lower()
    return mode if mode in MODE_CONDITIONS else "customer"


def chroma_filter(mode, extra: list[dict] | None = None) -> dict:
    """The Chroma `where` filter for a mode, optionally with extra conditions."""
    conditions = MODE_CONDITIONS[normalise_mode(mode)] + (extra or [])
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def is_allowed(metadata: dict, mode) -> bool:
    """The same rule in Python, for BM25 and as a second check after every search."""
    return all(metadata.get(key) == value
               for condition in MODE_CONDITIONS[normalise_mode(mode)]
               for key, value in condition.items())


# =====================================================================
# Section 3 - Open the database (from the manifest)
# =====================================================================

current = {"index_path": None}


# Common words that every chunk shares. Without this, BM25 rewards a chunk for
# matching "what", "is" and "the". "no" and "not" are kept: they change meaning ("No Claim Bonus").
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "have", "has", "had",
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "how", "why", "if", "then", "so", "there", "about", "any",
}


def tokenize(text: str) -> list[str]:
    """'Is the Batticaloa branch open on Saturday?' -> ['batticaloa', 'branch', 'open', 'saturday']."""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


def load_index(info_file: Path = config.INDEX_INFO_FILE, embeddings=None) -> dict:
    """Open the build named in index_info.json. Does nothing if it is already open.

    Called before every question, so a rebuild from Notebook 1 is picked up
    automatically. A whole new `current` is built and swapped in with one line,
    so a question never sees half of the old build and half of the new.
    """
    global current
    info_file = Path(info_file)
    if not info_file.exists():
        raise FileNotFoundError("No database found. Run notebooks/1_ingest.ipynb first.")

    info = json.loads(info_file.read_text(encoding="utf-8"))
    if info["index_path"] == current["index_path"]:
        return current  # this build is already open

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    index_path = Path(info["index_path"])
    if not index_path.is_absolute():
        index_path = config.ROOT / index_path

    vectorstore = Chroma(
        persist_directory=str(index_path),
        # The model comes from the manifest, so it always matches the stored vectors
        embedding_function=embeddings or OpenAIEmbeddings(model=info["embedding_model"]),
        collection_name=info["collection_name"],
    )

    # Keyword search needs all the chunks in memory
    data = vectorstore.get(include=["documents", "metadatas"])
    chunks = [Document(page_content=text, metadata=metadata)
              for text, metadata in zip(data["documents"], data["metadatas"])]

    current = {
        "index_path": info["index_path"],
        "info": info,
        "vectorstore": vectorstore,
        "chunks": chunks,
        "keyword_index": BM25Okapi([tokenize(c.page_content) for c in chunks]),
        "aliases": document_aliases(chunks),
    }
    print(f"Opened {info['index_path']} with {len(chunks)} chunks ({info['embedding_model']})")
    return current


# =====================================================================
# Section 5 - Keyword search and vector search
# =====================================================================

def vector_search(question: str, mode, k: int | None = None) -> list[Document]:
    """Chunks with a similar MEANING, inside what this mode may see."""
    k = k or config.SEARCH_RESULTS
    return current["vectorstore"].similarity_search(question, k=k, filter=chroma_filter(mode))


def keyword_search(question: str, mode, k: int | None = None) -> list[Document]:
    """Chunks that share the most WORDS with the question, inside what this mode may see."""
    k = k or config.SEARCH_RESULTS
    scores = current["keyword_index"].get_scores(tokenize(question))
    ranked = sorted(zip(scores, current["chunks"]), key=lambda pair: pair[0], reverse=True)
    results = [chunk for score, chunk in ranked if score > 0 and is_allowed(chunk.metadata, mode)]
    return results[:k]


# =====================================================================
# Section 6 - Hybrid search (Reciprocal Rank Fusion)
# =====================================================================

def hybrid_search(question: str, mode) -> list[Document]:
    """Run both searches and merge them by RANK, without repeats.

    BM25 scores (0-30) and vector distances (0-2) are on different scales,
    so we merge ranks instead: each list gives a chunk 1 / (60 + rank).
    A chunk found high by both searches comes first.
    """
    fused, by_id = {}, {}
    for results in (vector_search(question, mode), keyword_search(question, mode)):
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk.metadata["chunk_id"]
            by_id[chunk_id] = chunk
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)

    ordered = sorted(fused, key=fused.get, reverse=True)
    # Second check: nothing outside the mode may pass, even if a filter was wrong
    return [by_id[cid] for cid in ordered if is_allowed(by_id[cid].metadata, mode)]


# =====================================================================
# Section 7 - The reranker
# =====================================================================

@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once (~1 GB download on first use).

    Some versions of sentence-transformers return probabilities (0-1) for this
    model and some return raw scores. We test once, so scores are always 0-1.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(config.RERANKER_MODEL)
    probe = model.predict([
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Bananas are yellow."),
    ])
    returns_probabilities = all(0.0 <= float(score) <= 1.0 for score in probe)
    return model, returns_probabilities


def rerank(question: str, chunks: list[Document]) -> list[tuple[float, Document]]:
    """Score every chunk against the question (0-1). Returns (score, chunk) pairs, best first."""
    if not chunks:
        return []
    model, returns_probabilities = get_reranker()
    scores = np.asarray(model.predict([(question, c.page_content) for c in chunks]), dtype=float)
    if not returns_probabilities:
        scores = 1.0 / (1.0 + np.exp(-scores))   # sigmoid: raw score -> probability
    return sorted(zip(scores.tolist(), chunks), key=lambda pair: pair[0], reverse=True)


# =====================================================================
# Section 8 - Relevance check
# =====================================================================

def find_best_chunks(question: str, mode, best: int | None = None,
                     min_score: float | None = None) -> list[tuple[float, Document]]:
    """Search, rerank, and keep the best chunks - if the question is relevant at all.

    The check is on the BEST chunk: if even that one is below the cutoff, the documents
    do not cover this question, and an empty list is returned. The chatbot then says
    "I don't know" WITHOUT calling the LLM.
    If the best chunk passes, the top `best` chunks are kept, because one question can
    need several chunks (a policy rule + a premium table + an exclusion).
    """
    best = best or config.BEST_CHUNKS
    min_score = config.MIN_RELEVANCE_SCORE if min_score is None else min_score
    ranked = rerank(question, hybrid_search(question, mode))
    if not ranked or ranked[0][0] < min_score:
        return []
    return ranked[:best]


# =====================================================================
# Section 9 - Cross-references ("see Claims Overview - Section 4")
# =====================================================================

SEE_PATTERN = re.compile(r"\bsee\s+(?:the\s+|our\s+)?([^\n.;()]{3,80})", re.IGNORECASE)
FILE_PATTERN = re.compile(r"([\w-]+\.md)\b")


def document_aliases(chunks: list[Document]) -> dict[str, set[str]]:
    """Every name a document can be referred to by: file name, words of the file name, title, short title."""
    aliases = {}
    for chunk in chunks:
        m = chunk.metadata
        source = m["source_file"]
        if source in aliases:
            continue
        stem = Path(source).stem.lower()
        words = stem.replace("_", " ")
        title = str(m.get("title", "")).lower()
        short_title = re.split(r"\s[—–-]\s|\(|:", title)[0].strip()
        names = {source.lower(), stem, words, re.sub(r"\s\d{4}$", "", words), title, short_title}
        aliases[source] = {name for name in names if len(name) >= 8}
    return aliases


def referenced_files(text: str) -> list[str]:
    """Documents a chunk points to, e.g. 'see Claims Overview - Section 4' or 'see motor_premium_table_2026.md'.

    References use shortened names, so 'Claims Overview' must match claims_overview_procedure.md:
    we take the name part of the phrase (before a dash, comma or the word 'section')
    and match it against every alias in both directions.
    """
    found = [f for f in FILE_PATTERN.findall(text) if f in current["aliases"]]
    for phrase in SEE_PATTERN.findall(text):
        phrase = phrase.lower().strip()
        name = re.split(r"\s[—–-]\s|,|\bsection\b|\bpart\b", phrase)[0].strip(" `'\"")
        if len(name) < 6:
            continue
        for source, aliases in current["aliases"].items():
            if source not in found and any(name in alias or alias in phrase for alias in aliases):
                found.append(source)
    return found


def expand_cross_references(question: str, kept: list[tuple[float, Document]], mode,
                            limit: int | None = None) -> list[tuple[float, Document]]:
    """If a kept chunk says 'see X', also add X's best chunk for this question.

    Some answers need two documents (Q06, Q11). Search alone may find only one.
    The extra chunk still goes through the mode filter, so it can never be internal
    in Customer mode.
    """
    limit = config.MAX_CROSS_REFERENCES if limit is None else limit
    present = {chunk.metadata["source_file"] for _, chunk in kept}
    wanted = []
    for _, chunk in kept:
        for source in referenced_files(chunk.page_content):
            if source not in present and source not in wanted:
                wanted.append(source)

    extra = []
    for source in wanted[:limit]:
        results = current["vectorstore"].similarity_search(
            question, k=3, filter=chroma_filter(mode, [{"source_file": source}]))
        results = [r for r in results if is_allowed(r.metadata, mode)]
        if results:
            extra.append(rerank(question, results)[0])
    return kept + extra