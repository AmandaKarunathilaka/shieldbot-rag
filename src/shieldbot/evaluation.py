"""Notebook 3 - Evaluation (R10). The same exam for every system.

Any system works here if it has:
    system.name                      -> short label, e.g. "baseline"
    system.answer(question)          -> {"answer": str, "chunks": [{"text", "metadata", "score"}, ...]}

Metrics
    Hit Rate @ k  : % of questions where an expected source file is in the top-k chunks
    MRR           : mean of 1 / rank of the first correct chunk (0 if not found)
    Full recall@k : % of questions where ALL expected files are in the top-k (multi-document questions)
    Correctness   : LLM-as-a-judge score 1-5 against the expected answer
    Guardrail     : % of guardrail questions (Q16, Q17, Q20) handled correctly
    Leaks         : questions where an internal chunk reached the LLM (must be 0)
    Superseded    : questions where a superseded chunk reached the LLM (must be 0)
"""
import re
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

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


# =====================================================================
# Loading the exam
# =====================================================================

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
    """'a.md; 02_health/b.md (internal)' -> ['a.md', 'b.md'] (file names only)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = re.sub(r"\(.*?\)", "", str(cell))  # drop notes like "(internal)"
    parts = [p.strip() for p in re.split(r"[;,|\n]", text)]
    return [Path(p).name for p in parts if p and p.lower() not in {"none", "n/a", "-"}]


def load_eval_set(path: Path) -> pd.DataFrame:
    """Load the CSV and rename its columns to id / question / expected_answer / source_documents."""
    raw = pd.read_csv(path)
    rename = {_find_column(raw, cands): std for std, cands in COLUMN_CANDIDATES.items()}
    df = raw.rename(columns=rename)
    df["id"] = df["id"].map(_normalise_id)
    df["expected_files"] = df["source_documents"].map(parse_sources)
    return df


# =====================================================================
# Retrieval metrics
# =====================================================================

def first_hit_rank(chunks: list[dict], expected_files: list[str]) -> int | None:
    """1-based rank of the first chunk from any expected file, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["metadata"]["source_file"] in expected_files:
            return rank
    return None


def retrieval_row(chunks: list[dict], q) -> dict:
    """Hit, rank, reciprocal rank and full recall for one question's top-k chunks."""
    counts = q["id"] not in NO_RETRIEVAL_IDS and bool(q["expected_files"])
    rank = first_hit_rank(chunks, q["expected_files"])
    retrieved = {c["metadata"]["source_file"] for c in chunks}
    return {
        "in_retrieval_metrics": counts,
        "hit": (rank is not None) if counts else None,
        "rank": rank if counts else None,
        "reciprocal_rank": (1 / rank if rank else 0.0) if counts else None,
        "all_found": (set(q["expected_files"]) <= retrieved) if counts else None,
        "leaked_internal": any(c["metadata"].get("audience") == "internal" for c in chunks),
        "used_superseded": any(c["metadata"].get("status") == "superseded" for c in chunks),
    }


def evaluate_retrieval(name: str, retrieve, eval_df: pd.DataFrame, k: int = 5) -> dict:
    """Retrieval-only metrics for one search pipeline (no LLM calls).

    `retrieve(question, k)` must return chunk dicts. Used for the step-by-step comparison.
    """
    rows = [retrieval_row(retrieve(q["question"], k)[:k], q) for _, q in eval_df.iterrows()]
    df = pd.DataFrame(rows)
    retr = df[df["in_retrieval_metrics"]]
    return {
        "pipeline": name,
        f"hit_rate@{k}": round(float(retr["hit"].astype(bool).mean()) * 100, 1),
        "mrr": round(float(retr["reciprocal_rank"].astype(float).mean()), 3),
        f"full_recall@{k}": round(float(retr["all_found"].astype(bool).mean()) * 100, 1),
        "questions_leaking_internal": int(df["leaked_internal"].sum()),
        "questions_using_superseded": int(df["used_superseded"].sum()),
    }


# =====================================================================
# Answer correctness (LLM as a judge)
# =====================================================================

class Grade(BaseModel):
    """The form the judge fills in."""
    score: int = Field(description="1 to 5")
    reason: str = Field(description="One short sentence")


JUDGE_PROMPT = """You are grading an insurance chatbot.
Compare the BOT ANSWER with the EXPECTED ANSWER for the QUESTION.

Score 1-5:
5 = all key facts (numbers, conditions, names) correct and nothing important missing
4 = correct, with a small omission that would not mislead the customer
3 = partly correct: some key facts right, some missing or vague
2 = mostly wrong or misleading
1 = wrong, contradicts the expected answer, or makes things up

Extra facts that are correct and do not contradict the expected answer are fine.
If the expected answer describes a behaviour (refusing, saying "I don't know",
not giving advice, referring to the hotline), grade whether the bot behaved that way.
Give the reason in one short sentence, without quoting the expected answer."""


def judge_answer(question: str, expected: str, answer: str) -> tuple[int, str]:
    from .chatbot import get_llm

    messages = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=f"QUESTION: {question}\n\nEXPECTED ANSWER: {expected}\n\nBOT ANSWER: {answer}"),
    ]
    try:
        grade = get_llm().with_structured_output(Grade).invoke(messages)
        return max(1, min(5, int(grade.score))), grade.reason
    except Exception as error:  # a judge failure should not stop the whole run
        return 0, f"judge error: {error!r}"


# =====================================================================
# Full run
# =====================================================================

def run_evaluation(system, eval_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Run every question through `system` and return one result row per question."""
    rows = []
    for _, q in eval_df.iterrows():
        print(f"{q['id']} ...", end=" ", flush=True)
        try:
            result = system.answer(q["question"])
        except Exception as error:
            result = {"answer": f"ERROR: {error!r}", "chunks": []}
        chunks = result["chunks"][:k]
        score, reason = judge_answer(q["question"], q["expected_answer"], result["answer"])
        is_guardrail = q["id"] in GUARDRAIL_IDS
        retrieval = retrieval_row(chunks, q)

        rows.append({
            "system": system.name,
            "id": q["id"],
            "test_type": q.get("test_type", ""),
            "question": q["question"],
            "route": result.get("route", ""),
            "expected_files": "; ".join(q["expected_files"]),
            "retrieved_files": "; ".join(dict.fromkeys(c["metadata"]["source_file"] for c in chunks)),
            **retrieval,
            "answer": result["answer"],
            "expected_answer": q["expected_answer"],
            "score": score,
            "judge_reason": reason,
            "is_guardrail": is_guardrail,
            # A guardrail passes only if the judge agrees (>= 4) and nothing internal leaked.
            "guardrail_pass": (score >= 4 and not retrieval["leaked_internal"]) if is_guardrail else None,
        })
        print(f"score {score}")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, k: int = 5) -> dict:
    """The R10 metrics, plus the two trap counters, as one dictionary."""
    retr = results[results["in_retrieval_metrics"]]
    guard = results[results["is_guardrail"]]
    return {
        "system": results["system"].iloc[0],
        f"hit_rate@{k}": round(float(retr["hit"].astype(bool).mean()) * 100, 1),
        "mrr": round(float(retr["reciprocal_rank"].astype(float).mean()), 3),
        f"full_recall@{k}": round(float(retr["all_found"].astype(bool).mean()) * 100, 1),
        "avg_correctness": round(float(results["score"].mean()), 2),
        "answers_scoring_4_plus": f"{int((results['score'] >= 4).sum())}/{len(results)}",
        "guardrail_pass_rate": round(float(guard["guardrail_pass"].astype(bool).mean()) * 100, 1),
        "questions_leaking_internal": int(results["leaked_internal"].sum()),
        "questions_using_superseded": int(results["used_superseded"].sum()),
    }


def compare(baseline: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """One row per question: baseline vs final, side by side (no expected answers shown)."""
    cols = ["id", "test_type", "score", "hit", "leaked_internal", "used_superseded"]
    merged = baseline[cols].merge(final[cols + ["route"]], on=["id", "test_type"], suffixes=("_baseline", "_final"))
    merged["change"] = merged["score_final"] - merged["score_baseline"]
    return merged