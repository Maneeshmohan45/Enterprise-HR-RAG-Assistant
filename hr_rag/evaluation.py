"""Grounding checks, multi-metric evaluation, feedback logging, improvement hints."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .config import Settings, settings as default_settings

_GROUNDING_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "with", "this", "that", "it", "as", "be", "by", "from",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def lexical_grounding(answer: str, context: str) -> float:
    """Share of meaningful answer words that also appear in the context."""
    answer_words = {w for w in _normalize(answer).split() if w not in _GROUNDING_STOP and len(w) > 2}
    if not answer_words:
        return 0.0
    return len(answer_words & set(_normalize(context).split())) / len(answer_words)


_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
                 "thirty": 30}


def extract_numbers(text: str) -> set[str]:
    """Numbers in a text, ignoring [S1] markers and list numbering. Words count too ("five" = 5)."""
    text = re.sub(r"\[S\d+\]", " ", text)
    text = re.sub(r"(?m)^\s*(?:#+\s*)?\d+[.)]\s+", " ", text)
    numbers = {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", text)}
    numbers |= {str(_WORD_NUMBERS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in _WORD_NUMBERS}
    return numbers


def ungrounded_numbers(answer: str, context: str, question: str = "") -> list[str]:
    """Numbers the answer states that appear in neither the sources nor the question.

    Catches invented figures such as "26 weeks of maternity leave" when no source says it.
    """
    allowed = extract_numbers(context) | extract_numbers(question)
    return sorted(extract_numbers(answer) - allowed, key=float)


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product. Equals cosine similarity because embeddings are normalized."""
    return float(sum(x * y for x, y in zip(a, b)))


def semantic_grounding(answer: str, source_texts: list[str], embeddings) -> tuple[float, list[float]]:
    """Best cosine similarity between the answer and any single source.

    Each source is embedded on its own. Embedding the whole context at once would
    silently truncate it (MiniLM reads only ~256 tokens), so only the first
    source would count.
    """
    if not source_texts or not answer.strip():
        return 0.0, []
    vectors = embeddings.embed_documents([answer] + source_texts)
    scores = [cosine(vectors[0], v) for v in vectors[1:]]
    return max(scores), scores


def evaluate(question_embedding: list[float], answer: str, context: str, source_texts: list[str],
             sources: list[dict], status: str, citations_valid: bool, embeddings,
             cfg: Settings = default_settings, question: str = "") -> dict:
    lexical = lexical_grounding(answer, context)
    semantic, _ = semantic_grounding(answer, source_texts, embeddings)
    relevance = cosine(question_embedding, embeddings.embed_query(answer)) if answer.strip() else 0.0

    invented = ungrounded_numbers(answer, context, question)
    checks = {
        "answer_available": bool(answer.strip()),
        "has_sources": len(sources) > 0,
        "citations_valid": citations_valid,
        "lexically_grounded": lexical >= cfg.lexical_grounding_threshold,
        "semantically_grounded": semantic >= cfg.semantic_grounding_threshold,
        "answer_relevant": relevance >= cfg.answer_relevance_threshold,
        "numbers_grounded": not invented,
        "access_verified": status == "access_verified",
    }
    return {
        **checks,
        "ungrounded_numbers": invented,
        "lexical_grounding_score": round(lexical, 3),
        "semantic_grounding_score": round(semantic, 3),
        "answer_relevance_score": round(relevance, 3),
        "evaluation_score": round(sum(checks.values()) / len(checks) * 100, 2),
    }


def improvement_actions(report: dict, rating: int | None = None, helpful: bool | None = None) -> list[str]:
    actions = []
    if rating is not None and rating <= 2:
        actions.append("Review answer clarity, completeness, and usefulness")
    if helpful is False:
        actions.append("Inspect the retrieved evidence behind this answer")
    if not report["citations_valid"]:
        actions.append("Fix citation prompting and source-ID validation")
    if not report["has_sources"]:
        actions.append("Review retrieval and citation generation")
    if not report["lexically_grounded"]:
        actions.append("Review lexical grounding and context coverage")
    if not report["semantically_grounded"]:
        actions.append("Improve reranking/context selection for stronger semantic grounding")
    if not report["answer_relevant"]:
        actions.append("Review query rewriting, retrieval relevance, and answer focus")
    if not report.get("numbers_grounded", True):
        actions.append("Check the figures in the answer against the sources (possible invented numbers)")
    if not report["access_verified"]:
        actions.append("Investigate authorization/security invariant failure")
    return actions or ["No major improvement required for this query"]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_query(response: dict, cfg: Settings = default_settings) -> None:
    append_jsonl(cfg.logs_dir / "query_log.jsonl", {
        "query_id": response["query_id"],
        "time": datetime.now().isoformat(timespec="seconds"),
        "role": response["role"],
        "question": response["question"],
        "status": response["status"],
        "source_chunk_ids": [s["chunk_id"] for s in response["sources"]],
        "evaluation_score": response["evaluation"]["evaluation_score"] if response["evaluation"] else None,
    })


def record_feedback(response: dict, rating: int, helpful: bool, comment: str,
                    cfg: Settings = default_settings) -> dict:
    if not 1 <= rating <= 5:
        raise ValueError("rating must be between 1 and 5")
    evaluation = response["evaluation"] or {}
    record = {
        "query_id": response["query_id"],
        "question": response["question"],
        "rating": rating,
        "helpful": helpful,
        "comment": comment,
        "evaluation_score": evaluation.get("evaluation_score"),
        "improvement_actions": improvement_actions(evaluation, rating, helpful) if evaluation else [],
        "feedback_time": datetime.now().isoformat(timespec="seconds"),
    }
    append_jsonl(cfg.logs_dir / "feedback.jsonl", record)
    return record