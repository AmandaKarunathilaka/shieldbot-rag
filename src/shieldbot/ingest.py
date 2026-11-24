"""Step 1 - Smart ingestion (R1).

Load every knowledge-base document and turn its YAML front matter into metadata.
Each document becomes {"text": body, "metadata": {...}}.
"""
from datetime import date, datetime
from pathlib import Path

import frontmatter
import pandas as pd

REQUIRED_FIELDS = ["doc_id", "title", "product", "doc_type", "effective_date", "status", "audience"]


def clean_value(value):
    """Convert a front-matter value into a type ChromaDB can store (str, int, float, bool)."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()                 # 2026-01-01 -> "2026-01-01"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)  # ["a", "b"] -> "a, b"
    return str(value)


def first_heading(text: str) -> str:
    """Return the first '# ' heading, used as a fallback title."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def load_documents(kb_dir: Path) -> list[dict]:
    """Load every .md file (except README.md) as {'text': ..., 'metadata': {...}}."""
    documents = []
    for path in sorted(Path(kb_dir).rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue

        post = frontmatter.load(path, encoding="utf-8")
        if not post.metadata:
            print(f"No front matter, skipped: {path.name}")
            continue

        metadata = {key: clean_value(val) for key, val in post.metadata.items()}
        metadata.setdefault("title", first_heading(post.content) or path.stem)
        metadata["source_file"] = path.name
        metadata["folder"] = path.parent.name
        metadata["path"] = path.relative_to(kb_dir).as_posix()

        documents.append({"text": post.content.strip(), "metadata": metadata})
    return documents


def metadata_table(documents: list[dict]) -> pd.DataFrame:
    """One row per document, most useful columns first."""
    df = pd.DataFrame([d["metadata"] for d in documents])
    df["chars"] = [len(d["text"]) for d in documents]
    preferred = ["folder", "source_file", "doc_id", "title", "product", "doc_type",
                 "effective_date", "status", "audience", "chars"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def check_documents(documents: list[dict]) -> None:
    """Fail loudly if the knowledge base is not what the filters expect."""
    df = metadata_table(documents)
    missing = {
        d["metadata"]["source_file"]: [f for f in REQUIRED_FIELDS if not d["metadata"].get(f)]
        for d in documents
    }
    missing = {k: v for k, v in missing.items() if v}

    assert not missing, f"Docs missing required fields: {missing}"
    assert len(documents) == 26, f"Expected 26 docs, got {len(documents)}"
    assert (df["status"] == "superseded").sum() == 1, "Expected 1 superseded doc"
    assert (df["audience"] == "internal").sum() == 2, "Expected 2 internal docs"
    print("Step 1 checks passed: 26 docs, 1 superseded, 2 internal, no missing fields")
    