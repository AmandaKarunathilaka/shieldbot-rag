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
CHAT_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None   # empty -> OpenAI default

# ---------- Chatbot search settings (Notebook 2) ----------
SEARCH_RESULTS = 20        # chunks each search (BM25, vector) finds
RRF_K = 60                 # Reciprocal Rank Fusion constant: score = 1 / (RRF_K + rank)
BEST_CHUNKS = 5            # chunks the reranker keeps for the LLM
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~90 MB, fast on CPU
MIN_RELEVANCE_SCORE = 0.001  # best chunk must reach this (0-1); calibrated in Notebook 2 (Section 8) and Notebook 3 (Q11)
OUT_OF_SCOPE_OVERRIDE_SCORE = 0.5  # a chunk this relevant overrules an "out of scope" guardrail call
MAX_CROSS_REFERENCES = 2   # extra documents added when a chunk says "see ..."

# ---------- Chatbot limits ----------
MAX_QUESTION_LENGTH = 1000
MAX_HISTORY_MESSAGES = 10
LLM_TIMEOUT = 30           # seconds
LLM_MAX_RETRIES = 2

# ---------- Contact details used in refusals and hand-offs ----------
# Check these against 00_company/contact_channels.md
HOTLINE = os.getenv("SHIELD_HOTLINE", "0117 000 111")
CARE_EMAIL = os.getenv("SHIELD_CARE_EMAIL", "care@ceylonshield.lk")

# ---------- Gradio logins (class demo only; a real product needs proper auth) ----------
USERS = [
    ("customer", os.getenv("CUSTOMER_PASSWORD", "customer123")),
    ("staff", os.getenv("STAFF_PASSWORD", "staff123")),
]