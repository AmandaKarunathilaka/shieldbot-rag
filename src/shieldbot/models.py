"""LLM client and embedding model, each loaded once and reused.

`lru_cache` makes the first call create the object and every later call
return the same one, so the ~130 MB embedding model is loaded only once.
"""
import json
import re
from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def get_llm():
    from openai import OpenAI  # imported here so modules load fast without the key

    return OpenAI(base_url=config.OPENAI_BASE_URL)  # reads OPENAI_API_KEY from env


def chat(messages: list[dict], temperature: float = 0.0, json_mode: bool = False) -> str:
    """Send chat messages to the LLM and return the reply text."""
    kwargs = {"model": config.LLM_MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = get_llm().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Parse a JSON reply, tolerating ```json fences some models add."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed chunks. Normalised, so a dot product equals cosine similarity."""
    return get_embedder().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )


def embed_query(question: str) -> np.ndarray:
    """Embed one question, with the bge query instruction."""
    return get_embedder().encode(
        config.BGE_QUERY_INSTRUCTION + question, normalize_embeddings=True
    )