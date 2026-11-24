"""Project paths and settings, shared by every module.

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
load_dotenv(ROOT / ".env")

# Folders
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "ceylon_shield_knowledge_base"
EVAL_CSV = DATA_DIR / "shieldbot_evaluation_set.csv"
STORAGE_DIR = ROOT / "storage"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None  # empty -> OpenAI default
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge v1.5 retrieves a little better when short queries carry this instruction.
# Passages (chunks) are embedded without it.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "