"""All settings in one place, shared by both notebooks and the Gradio app.

Paths are found from this file's location, not from the current folder,
so imports work the same from notebooks/, the project root or app.py.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


def find_root(start: Path) -> Path:
    """Walk up from `start` until the folder that holds pyproject.toml."""
    path = start
    while not (path / "pyproject.toml").exists() and path != path.parent:
        path = path.parent
    return path


ROOT = find_root(Path(__file__).resolve().parent)
load_dotenv(ROOT / ".env", override=True)

# ---------- Folders ----------
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "ceylon_shield_knowledge_base"
EVAL_CSV = DATA_DIR / "shieldbot_evaluation_set.csv"
STORAGE_DIR = ROOT / "storage"              # one sub-folder per database build
INDEX_INFO_FILE = STORAGE_DIR / "index_info.json"  # the manifest: which build is current
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# ---------- Ingestion (Notebook 1) ----------
EMBEDDING_MODEL = "text-embedding-3-large"  # the chatbot reads this from index_info.json
COLLECTION_NAME = "ceylon_shield"
CHUNK_SIZE = 1000          # characters; a table is never cut, even if it is longer
BUILDS_TO_KEEP = 2         # the newest build is used, the one before it is a backup

# ---------- Chatbot (Notebook 2) ----------
CHAT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None   # empty -> OpenAI default

# ---------- Step 2 baseline only (kept until the evaluation notebook in Step 10) ----------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
LLM_MODEL = CHAT_MODEL