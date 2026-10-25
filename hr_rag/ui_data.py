"""Helpers for the Streamlit UI that need no Streamlit, so they can be unit-tested offline.

* reading ``logs/query_log.jsonl`` and ``logs/feedback.jsonl``
* summarising them for the Model Evaluation page (real numbers only)
* turning exceptions into short, safe messages (API keys are never shown)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Which of the quality checks in ``evaluation.evaluate`` are shown, in order.
CHECK_LABELS = {
    "answer_available": "Answer produced",
    "has_sources": "Has sources",
    "citations_valid": "Citations valid",
    "lexically_grounded": "Lexical grounding",
    "semantically_grounded": "Semantic grounding",
    "answer_relevant": "Answer relevance",
    "numbers_grounded": "Numbers grounded",
    "access_verified": "Access verified",
}


# ---------------------------------------------------------------------------
# Log reading
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file. A missing file gives []; unreadable lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


CLASS_LABELS = {"NORMAL_GROUNDED": "Normal grounded answer", "SECURITY_REFUSAL": "Security refusal",
                "ACCESS_RESTRICTED": "Restricted access", "OUT_OF_DOMAIN": "Out of domain",
                "NOT_FOUND": "Not found in the documents", "LEGACY": "Before response classes"}


def class_integrity(rows: list[dict]) -> dict:
    """For refusal / not-found classes: how many rows passed their own integrity checks.

    Normal answers are judged by the quality score instead, so they are not listed here. Rows without
    a class (older log rows) are never counted.
    """
    out: dict[str, dict] = {}
    for row in rows:
        cls, passed = row.get("response_class"), row.get("class_passed")
        if cls and cls != "NORMAL_GROUNDED" and passed is not None:
            entry = out.setdefault(cls, {"passed": 0, "total": 0})
            entry["total"] += 1
            entry["passed"] += bool(passed)
    return out


def summarize_queries(rows: list[dict]) -> dict:
    """Aggregate the query log. Anything not recorded for a row is left out, never guessed."""
    scores = [r.get("evaluation_score") for r in rows]
    timed = [r for r in rows if (r.get("timings") or {}).get("total_s") is not None]
    used = [r for r in rows if (r.get("usage") or {}).get("output_tokens") is not None]
    with_checks = [r for r in rows if isinstance(r.get("checks"), dict)]

    pass_rates = {}
    for key in CHECK_LABELS:
        values = [r["checks"][key] for r in with_checks if key in r["checks"]]
        if values:
            pass_rates[key] = sum(bool(v) for v in values) / len(values)

    by_style = {}
    for style in sorted({r.get("style") for r in used if r.get("style")}):
        group = [r for r in used if r.get("style") == style]
        by_style[style] = {
            "n": len(group),
            "avg_input_tokens": _mean([r["usage"].get("input_tokens") for r in group]),
            "avg_output_tokens": _mean([r["usage"].get("output_tokens") for r in group]),
            "avg_total_s": _mean([(r.get("timings") or {}).get("total_s") for r in group]),
        }

    return {
        "n_queries": len(rows),
        "n_scored": sum(s is not None for s in scores),
        "avg_score": _mean(scores),
        "n_timed": len(timed),
        "avg_total_s": _mean([r["timings"]["total_s"] for r in timed]),
        "avg_retrieval_s": _mean([r["timings"].get("retrieval_s") for r in timed]),
        "avg_generation_s": _mean([r["timings"].get("generation_s") for r in timed]),
        "n_with_usage": len(used),
        "avg_input_tokens": _mean([r["usage"].get("input_tokens") for r in used]),
        "avg_output_tokens": _mean([r["usage"].get("output_tokens") for r in used]),
        "status_counts": dict(Counter(r.get("status", "unknown") for r in rows)),
        "class_counts": dict(Counter(r.get("response_class") or "LEGACY" for r in rows)),
        "class_integrity": class_integrity(rows),
        "n_with_checks": len(with_checks),
        "check_pass_rates": pass_rates,
        "by_style": by_style,
    }


def summarize_feedback(rows: list[dict]) -> dict:
    ratings = [r.get("rating") for r in rows if isinstance(r.get("rating"), (int, float))]
    return {
        "n": len(rows),
        "helpful": sum(1 for r in rows if r.get("helpful") is True),
        "not_helpful": sum(1 for r in rows if r.get("helpful") is False),
        "avg_rating": _mean(ratings),
    }


def query_table_rows(rows: list[dict], show_questions: bool, limit: int = 200) -> list[dict]:
    """Newest-first rows for the log table. Questions are hidden unless allowed
    (they can mention confidential records)."""
    table = []
    for r in reversed(rows[-limit:]):
        usage, timings = r.get("usage") or {}, r.get("timings") or {}
        question = " ".join(str(r.get("question", "")).split())
        table.append({
            "Time": str(r.get("time", "")).replace("T", " "),
            "Role": r.get("role", ""),
            "Style": r.get("style") or "n/a",
            "Status": r.get("status", ""),
            "Score %": r.get("evaluation_score"),
            "Tokens in": usage.get("input_tokens"),
            "Tokens out": usage.get("output_tokens"),
            "Seconds": timings.get("total_s"),
            "Question": (question[:110] + ("..." if len(question) > 110 else ""))
            if show_questions else "(hidden for this role)",
        })
    return table


# ---------------------------------------------------------------------------
# Safe error messages
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(?:nvapi|sk-ant|sk)-[A-Za-z0-9_\-\*\.]{4,}"),
    re.compile(r"(?i)(api[ _-]?key[^:=\n]{0,40}[:=]\s*)\S+"),
    re.compile(r"(?i)(bearer\s+)\S+"),
]


def redact(text: str) -> str:
    """Remove anything that looks like an API key before showing an error."""
    text = _SECRET_PATTERNS[0].sub("[redacted]", str(text))
    text = _SECRET_PATTERNS[1].sub(r"\1[redacted]", text)
    return _SECRET_PATTERNS[2].sub(r"\1[redacted]", text)


def friendly_error(error: Exception) -> tuple[str, str]:
    """Return ``(message_for_the_user, redacted_detail)`` for an exception from the pipeline."""
    detail = redact(f"{type(error).__name__}: {error}")
    text = str(error).lower()
    kind = type(error).__name__

    if isinstance(error, PermissionError):
        return str(error) or "You do not have permission to do that.", detail
    if isinstance(error, ValueError):
        if any(w in text for w in ("api_key", "is missing", "unsupported llm_provider")):
            return ("The language model is not configured. Add your key to .env "
                    "(see .env.example) and restart the app."), detail
        return str(error), detail            # e.g. empty or too-long question
    if kind == "LLMError":
        if any(w in text for w in ("401", "403", "unauthorized", "authentication", "api key",
                                   "invalid key", "incorrect api")):
            return ("The language model provider rejected the API key. "
                    "Check the key in .env and restart the app."), detail
        if any(w in text for w in ("429", "rate limit", "quota")):
            return "The language model is rate-limited right now. Wait a moment and try again.", detail
        if any(w in text for w in ("timed out", "timeout", "connection", "network")):
            return "Could not reach the language model. Check your connection and try again.", detail
        if "cut off" in text or "empty" in text or "reasoning notes" in text:
            return ("The model did not return a usable answer. Try again, or see the details "
                    "below for the .env setting to check."), detail
        return "The language model could not answer this time. Please try again.", detail
    if isinstance(error, RuntimeError) and "vector store is empty" in text:
        return "The document index is empty. Build it first with:  python ingest.py", detail
    return "Something unexpected went wrong. Please try again.", detail
