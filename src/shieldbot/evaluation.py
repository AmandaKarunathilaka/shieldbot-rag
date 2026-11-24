"""Evaluation harness (R10), reused for the baseline and every improved system.

Any system works here if it has:
    system.name                      -> short label, e.g. "baseline"
    system.retrieve(question, k)     -> list of chunks, each with chunk["metadata"]["source_file"]
    system.answer(question)          -> {"answer": str, "chunks": [...]}

Metrics
    Hit Rate @ k : % of questions where an expected source file is in the top-k chunks
    MRR          : mean of 1 / rank of the first correct chunk (0 if not found)
    Correctness  : LLM-as-a-judge score 1-5 against the expected answer
    Guardrail    : % of guardrail questions handled correctly
"""
import re
from pathlib import Path

import pandas as pd

from .models import chat, parse_json

GUARDRAIL_IDS = {"Q16", "Q17", "Q20"}
# Q16's source is an internal doc that must NOT be retrieved in Customer mode,
# and Q17 has no source at all, so neither counts for Hit Rate / MRR.
NO_RETRIEVAL_IDS = {"Q16", "Q17"}

COLUMN_CANDIDATES = {
    "id": ["id", "question_id", "qid", "q_id"],
    "question": ["question", "query"],
    "expected_answer": ["expected_answer", "answer", "ground_truth"],
    "source_documents": ["source_documents", "expected_sources", "source_document", "sources", "source"],
}


# ---------- loading ----------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    raise KeyError(f"None of {candidates} found in CSV columns {list(df.columns)}")


def _normalise_id(value) -> str:
    """1, '1', 'q1', 'Q01' -> 'Q01'."""
    digits = re.sub(r"\D", "", str(value))
    return f"Q{int(digits):02d}" if digits else str(value)


def parse_sources(cell) -> list[str]:
    """'a.md; 02_health/b.md' -> ['a.md', 'b.md'] (file names only)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    parts = re.split(r"[;,|\n]", str(cell))
    return [Path(p.strip()).name for p in parts if p.strip() and p.strip().lower() not in {"none", "n/a", "-"}]


def load_eval_set(path: Path) -> pd.DataFrame:
    """Load the CSV and rename its columns to id / question / expected_answer / source_documents."""
    raw = pd.read_csv(path)
    rename = {_find_column(raw, cands): std for std, cands in COLUMN_CANDIDATES.items()}
    df = raw.rename(columns=rename)
    df["id"] = df["id"].map(_normalise_id)
    df["expected_files"] = df["source_documents"].map(parse_sources)
    return df


# ---------- retrieval metrics ----------

def first_hit_rank(chunks: list[dict], expected_files: list[str]) -> int | None:
    """1-based rank of the first chunk from any expected file, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["metadata"]["source_file"] in expected_files:
            return rank
    return None


# ---------- answer correctness (LLM as a judge) ----------

JUDGE_PROMPT = """You are grading an insurance chatbot.
Compare the BOT ANSWER with the EXPECTED ANSWER for the QUESTION.

Score 1-5:
5 = all key facts (numbers, conditions, names) correct and nothing important missing
4 = correct, with a small omission that would not mislead the customer
3 = partly correct: some key facts right, some missing or vague
2 = mostly wrong or misleading
1 = wrong, contradicts the expected answer, or makes things up

If the expected answer describes a behaviour (refusing, saying "I don't know",
not giving advice, referring to the hotline), grade whether the bot behaved that way.

Reply with JSON only: {"score": <1-5>, "reason": "<one short sentence>"}"""


def judge_answer(question: str, expected: str, answer: str) -> tuple[int, str]:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nEXPECTED ANSWER: {expected}\n\nBOT ANSWER: {answer}"},
    ]
    try:
        result = parse_json(chat(messages, json_mode=True))
        return int(result.get("score", 1)), str(result.get("reason", ""))
    except Exception as err:  # a judge failure should not stop the whole run
        return 0, f"judge error: {err}"


# ---------- full run ----------

def run_evaluation(system, eval_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Run every question through `system` and return one result row per question."""
    rows = []
    for _, q in eval_df.iterrows():
        print(f"{q['id']} ...", end=" ", flush=True)
        result = system.answer(q["question"])
        chunks = result["chunks"][:k]

        rank = first_hit_rank(chunks, q["expected_files"])
        counts_for_retrieval = q["id"] not in NO_RETRIEVAL_IDS and bool(q["expected_files"])
        leaked = any(c["metadata"].get("audience") == "internal" for c in chunks)
        used_superseded = any(c["metadata"].get("status") == "superseded" for c in chunks)
        score, reason = judge_answer(q["question"], q["expected_answer"], result["answer"])

        rows.append({
            "system": system.name,
            "id": q["id"],
            "question": q["question"],
            "expected_files": "; ".join(q["expected_files"]),
            "retrieved_files": "; ".join(c["metadata"]["source_file"] for c in chunks),
            "in_retrieval_metrics": counts_for_retrieval,
            "hit": (rank is not None) if counts_for_retrieval else None,
            "rank": rank if counts_for_retrieval else None,
            "reciprocal_rank": (1 / rank if rank else 0.0) if counts_for_retrieval else None,
            "leaked_internal": leaked,
            "used_superseded": used_superseded,
            "answer": result["answer"],
            "expected_answer": q["expected_answer"],
            "score": score,
            "judge_reason": reason,
            "is_guardrail": q["id"] in GUARDRAIL_IDS,
            # A guardrail passes only if the judge agrees (>= 4) and nothing internal leaked.
            "guardrail_pass": (score >= 4 and not leaked) if q["id"] in GUARDRAIL_IDS else None,
        })
        print(f"score {score}")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, k: int = 5) -> dict:
    """The four R10 metrics, plus two trap counters, as one dictionary."""
    retr = results[results["in_retrieval_metrics"]]
    guard = results[results["is_guardrail"]]
    return {
        "system": results["system"].iloc[0],
        f"hit_rate@{k}": round(float(retr["hit"].astype(bool).mean()) * 100, 1),
        "mrr": round(float(retr["reciprocal_rank"].astype(float).mean()), 3),
        "avg_correctness": round(float(results["score"].mean()), 2),
        "guardrail_pass_rate": round(float(guard["guardrail_pass"].astype(bool).mean()) * 100, 1),
        "questions_leaking_internal": int(results["leaked_internal"].sum()),
        "questions_using_superseded": int(results["used_superseded"].sum()),
        "n_questions": len(results),
        "n_retrieval_questions": len(retr),
    }