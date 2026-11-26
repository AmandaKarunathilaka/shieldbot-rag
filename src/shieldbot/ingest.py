"""Notebook 1 - Ingestion pipeline.

    Load (safely) -> Clean -> Chunk (headings + whole tables + context header)
    -> Check -> Build a new database -> Switch with a manifest -> Remove old builds

Every document is indexed, including superseded and internal ones.
The chatbot decides at SEARCH time what each mode may see (R3, R4), so Staff mode
can still reach internal documents and nothing has to be re-embedded.
"""
import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path

import frontmatter
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from . import config

REQUIRED_FIELDS = ["doc_id", "title", "product", "doc_type", "effective_date", "status", "audience"]


# =====================================================================
# 1. Load the files safely (R1)
# =====================================================================

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


def load_documents(folder: Path) -> tuple[list[Document], list[str]]:
    """Read every .md file. A bad file is skipped and reported, never fatal.

    Returns (documents, skipped). Each document's metadata is its YAML front matter
    plus source_file, folder and path.
    """
    documents, skipped = [], []

    for path in sorted(Path(folder).rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue  # a folder README is not company content

        try:
            post = frontmatter.load(path, encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append(f"{path.name} (not UTF-8)")
            continue
        except Exception as error:  # broken YAML front matter
            skipped.append(f"{path.name} ({type(error).__name__})")
            continue

        if not post.metadata:
            skipped.append(f"{path.name} (no front matter)")
            continue

        metadata = {key: clean_value(val) for key, val in post.metadata.items()}
        metadata.setdefault("title", first_heading(post.content) or path.stem)
        metadata["source_file"] = path.name
        metadata["folder"] = path.parent.name
        metadata["path"] = path.relative_to(folder).as_posix()

        documents.append(Document(page_content=post.content, metadata=metadata))

    return documents, skipped


def metadata_table(documents: list[Document]) -> pd.DataFrame:
    """One row per document, most useful columns first."""
    df = pd.DataFrame([d.metadata for d in documents])
    df["chars"] = [len(d.page_content) for d in documents]
    preferred = ["folder", "source_file", "doc_id", "title", "product", "doc_type",
                 "effective_date", "status", "audience", "chars"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def check_documents(documents: list[Document]) -> None:
    """Stop if the knowledge base is not what the filters expect."""
    df = metadata_table(documents)
    missing = {
        d.metadata["source_file"]: [f for f in REQUIRED_FIELDS if not d.metadata.get(f)]
        for d in documents
    }
    missing = {k: v for k, v in missing.items() if v}

    assert not missing, f"Docs missing required fields: {missing}"
    assert len(documents) == 26, f"Expected 26 docs, got {len(documents)}"
    assert (df["status"] == "superseded").sum() == 1, "Expected 1 superseded doc"
    assert (df["audience"] == "internal").sum() == 2, "Expected 2 internal docs"
    print("Document checks passed: 26 docs, 1 superseded, 2 internal, no missing fields")


# =====================================================================
# 2. Clean the text
# =====================================================================

def clean_text(text: str) -> str:
    """Remove signature lines, divider lines and extra blank lines.

    Table separator rows like |---|---| are kept: they start with '|'.
    """
    text = re.sub(r"_{5,}", "", text)                                # signature lines
    text = re.sub(r"^\s*(-{3,}|\*{3,})\s*$", "", text, flags=re.MULTILINE)  # --- dividers
    text = re.sub(r"\n{3,}", "\n\n", text)                           # extra blank lines
    return text.strip()


# =====================================================================
# 3. Smart chunks (R2): headings + whole tables + context header
# =====================================================================

# strip_headers=False keeps the heading text inside the chunk
heading_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)

TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")   # the |---|---| row under a table header


def is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def split_blocks(text: str) -> list[str]:
    """Split a section into blocks: paragraphs, list runs, and WHOLE tables.

    The heading splitter marks a blank line as a line ending with two spaces,
    so a line ending in "  " (or an empty line) closes the current block.
    Every run of lines that start with '|' becomes one block, so a table is never cut.
    """
    blocks, current, in_table = [], [], False

    def flush():
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        table_line = is_table_line(line)

        if table_line != in_table:      # entering or leaving a table
            flush()
            in_table = table_line
        if not line.strip():            # blank line ends a paragraph
            flush()
            continue

        current.append(line)
        if raw_line.endswith("  ") and not in_table:
            flush()                     # end of a paragraph
    flush()
    return [b for b in blocks if b]


def split_long_text(block: str, size: int) -> list[str]:
    """Cut a long NON-table block so each part fits `size`: between lines first,
    then between sentences for a single very long line. Tables are returned whole."""
    if len(block) <= size or is_table_line(block):
        return [block]

    units = []
    for line in block.splitlines():
        if len(line) <= size:
            units.append(line)
        else:
            units.extend(re.split(r"(?<=[.!?])\s+", line))

    pieces, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > size:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def is_heading_only(text: str) -> bool:
    return all(line.lstrip().startswith("#") for line in text.splitlines() if line.strip())


def pack_blocks(blocks: list[str], size: int) -> list[str]:
    """Join blocks into pieces of up to `size` characters, never splitting a block.

    A heading is never left alone: it always joins the block below it,
    so a table stays together with its heading even if that makes the piece long.
    """
    pieces, current = [], ""
    for block in blocks:
        for part in split_long_text(block, size):
            too_long = current and len(current) + len(part) + 2 > size
            if too_long and not is_heading_only(current):
                pieces.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def real_text_length(text: str) -> int:
    """Characters that are not heading lines, to spot title-only chunks."""
    return sum(len(line.strip()) for line in text.splitlines() if not line.lstrip().startswith("#"))


def context_header(doc: Document, section_path: str) -> str:
    """The label at the top of every chunk, so a chunk always says what it is about.

    'Product' is what separates the health waiting period from the life one.
    """
    m = doc.metadata
    return (f"Document: {m['title']}\n"
            f"Product: {m['product']} | Type: {m['doc_type']} | Effective: {m['effective_date']}\n"
            f"Section: {section_path}\n\n")


def make_chunks(doc: Document, size: int = config.CHUNK_SIZE) -> list[Document]:
    """Cut one document into chunks, and add a context header to each one."""
    chunks = []
    sections = heading_splitter.split_text(clean_text(doc.page_content))

    for section in sections:
        path = [section.metadata[h] for h in ("h1", "h2", "h3") if section.metadata.get(h)]
        heading = path[-1] if path else doc.metadata["title"]
        section_path = " > ".join(path) or doc.metadata["title"]
        level = max(len(path), 1)

        pieces = pack_blocks(split_blocks(section.page_content), size)
        for i, piece in enumerate(pieces):
            # A long section is cut into several pieces. Later pieces repeat the
            # heading, so a table always travels together with its heading.
            if i > 0:
                piece = f"{'#' * level} {heading} (continued)\n\n{piece}"

            # Skip chunks that are only a title, e.g. "# Health Shield Plus" on its own.
            # Short real content ("Closed on Sunday.") is kept: in insurance it can matter.
            if real_text_length(piece) == 0:
                continue

            chunk_id = f"{doc.metadata['doc_id']}::{len(chunks):03d}"
            chunks.append(Document(
                page_content=context_header(doc, section_path) + piece,
                metadata={
                    **doc.metadata,
                    "section": heading,
                    "section_path": section_path,
                    "has_table": any(is_table_line(line) for line in piece.splitlines()),
                    "chunk_id": chunk_id,
                },
            ))
    return chunks


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = []
    for doc in documents:
        chunks.extend(make_chunks(doc))
    return chunks


# =====================================================================
# 4. Check the chunks before we pay for embeddings
# =====================================================================

def check_chunks(chunks: list[Document]) -> pd.DataFrame:
    """Stop before embedding if any chunk is broken. Returns a size summary."""
    if not chunks:
        raise ValueError("No chunks were created. Check config.KB_DIR.")

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk ids"

    # Every chunk that holds table rows must also hold the table's header row.
    # If a table had been cut, the second half would have rows but no |---| row.
    broken = [
        c.metadata["chunk_id"] for c in chunks
        if c.metadata["has_table"]
        and not any(TABLE_SEPARATOR.match(line.strip()) for line in c.page_content.splitlines())
    ]
    assert not broken, f"Tables were cut (rows without a header) in: {broken}"

    sizes = pd.Series([len(c.page_content) for c in chunks])
    summary = pd.DataFrame({
        "chunks": [len(chunks)],
        "with_table": [sum(c.metadata["has_table"] for c in chunks)],
        "smallest": [sizes.min()],
        "median": [int(sizes.median())],
        "largest": [sizes.max()],
    })
    print("Chunk checks passed: unique ids, every table chunk has its header row")
    return summary


# =====================================================================
# 5. Build the new database safely (in its own folder)
# =====================================================================

def build_index(chunks: list[Document], storage_dir: Path = config.STORAGE_DIR,
                embedding_model: str = config.EMBEDDING_MODEL,
                collection_name: str = config.COLLECTION_NAME, embeddings=None):
    """Embed the chunks into a NEW timestamped folder. The old build is never touched.

    `embeddings` can be passed in for tests; by default it is OpenAI.
    Returns (build_folder, vectorstore).
    """
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    build_folder = Path(storage_dir) / datetime.now().strftime("build_%Y%m%d_%H%M%S")
    embeddings = embeddings or OpenAIEmbeddings(model=embedding_model, max_retries=5)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=[c.metadata["chunk_id"] for c in chunks],
        persist_directory=str(build_folder),
        collection_name=collection_name,
    )
    return build_folder, vectorstore


# =====================================================================
# 6. Switch to the new database (the manifest)
# =====================================================================

def switch_to_build(vectorstore, build_folder: Path, expected_count: int,
                    info_file: Path = config.INDEX_INFO_FILE,
                    embedding_model: str = config.EMBEDDING_MODEL,
                    collection_name: str = config.COLLECTION_NAME) -> dict:
    """Point index_info.json at the new build, but only if every chunk was saved.

    The file is written to .tmp first and then renamed: a rename happens all at once,
    so the chatbot can never read a half-written manifest.
    """
    saved_count = vectorstore._collection.count()
    if saved_count != expected_count:
        raise ValueError(f"Expected {expected_count} chunks but saved {saved_count}. "
                         "Keeping the old database.")

    build_folder = Path(build_folder).resolve()
    try:  # relative to the project root, so the manifest works on any computer
        index_path = build_folder.relative_to(config.ROOT).as_posix()
    except ValueError:
        index_path = build_folder.as_posix()

    index_info = {
        "index_path": index_path,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "chunk_count": saved_count,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    info_file = Path(info_file)
    temporary_file = info_file.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(index_info, indent=2), encoding="utf-8")
    temporary_file.replace(info_file)
    return index_info


def remove_old_builds(storage_dir: Path = config.STORAGE_DIR,
                      keep: int = config.BUILDS_TO_KEEP) -> tuple[int, int]:
    """Delete all but the newest `keep` builds. Returns (found, removed)."""
    all_builds = sorted(Path(storage_dir).glob("build_*"))
    old_builds = all_builds[:-keep] if keep else all_builds
    for old_build in old_builds:
        shutil.rmtree(old_build, ignore_errors=True)  # Windows may still hold a file open
    return len(all_builds), len(old_builds)