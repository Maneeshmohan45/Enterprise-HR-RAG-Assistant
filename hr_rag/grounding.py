"""Answer-side checks that need only the answer, the sources and the question.

``numeric_issues`` is the numeric gate: an answer may state only numbers that the sources give (or that
the question already contained), and a number must keep the time unit the sources give it. It withholds
invented figures ("26 weeks"), calculated results (3150 for an arithmetic question), and unit swaps
("7 weeks" when the policy says 7 days). It is deterministic: no model, no threshold.

``question_echo_anomaly`` is telemetry (below).

Answer-side anomaly telemetry.

Step 1 contains only ``question_echo_anomaly``: a SECONDARY signal that notices when an answer's
content words come from the question instead of the sources. It is recorded and shown in analysis
only. It never changes, withholds or rejects an answer: repeating words from the question is
normal, and a single heuristic must not decide what a user sees. The primary defences are the
request guard (decline clear injection), the Phase 1 filter and, later, the grounding gate which
judges answers only by how well the SOURCES support them.
"""
from __future__ import annotations

import re

from . import citations
from .evaluation import _GROUNDING_STOP, _WORD_NUMBERS, _normalize, extract_numbers, ungrounded_numbers

MIN_QUESTION_ONLY_TERMS = 4     # telemetry thresholds; provisional until calibrated on real answers
MIN_QUESTION_ONLY_RATIO = 0.25


def _content_words(text: str) -> set[str]:
    text = citations.MARKER.sub(" ", text)
    return {w for w in _normalize(text).split() if w not in _GROUNDING_STOP and len(w) > 2}


def question_echo_anomaly(question: str, answer: str, source_texts: list[str]) -> dict:
    """Count answer words that appear in the question but in none of the sources.

    Returns ``{"terms": int, "ratio": float, "flagged": bool}`` (counts only, no words, so a log
    never copies question or source text). Informational: callers must not act on ``flagged``.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return {"terms": 0, "ratio": 0.0, "flagged": False}
    source_words: set[str] = set()
    for text in source_texts:
        source_words |= _content_words(text)
    question_only = (answer_words & _content_words(question)) - source_words
    ratio = len(question_only) / len(answer_words)
    return {"terms": len(question_only), "ratio": round(ratio, 3),
            "flagged": len(question_only) >= MIN_QUESTION_ONLY_TERMS and ratio >= MIN_QUESTION_ONLY_RATIO}


# ---------------------------------------------------------------------------
# Numeric gate
# ---------------------------------------------------------------------------
_TIME_UNITS = {"day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
               "year": "year", "years": "year", "hour": "hour", "hours": "hour", "minute": "minute",
               "minutes": "minute"}
_UNIT_SKIP = frozenset({"working", "calendar", "business", "paid", "additional", "consecutive", "total", "full",
                        "half", "extra", "more", "per", "of"})
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")


def _unit_pairs(text: str) -> set[tuple[str, str]]:
    """(number, time unit) pairs such as ("7", "day") for "7 Days" or "seven working days"."""
    tokens = _TOKEN.findall(citations.MARKER.sub(" ", text.lower()))
    pairs: set[tuple[str, str]] = set()
    for i, token in enumerate(tokens):
        number = token if token[0].isdigit() else (str(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else None)
        if number is None:
            continue
        for following in tokens[i + 1:i + 4]:
            if following in _TIME_UNITS:
                pairs.add((number, _TIME_UNITS[following]))
                break
            if following not in _UNIT_SKIP:
                break
    return pairs


def numeric_issues(answer: str, source_text: str, question: str = "") -> list[dict]:
    """Numbers in the answer that the sources do not support.

    Each issue is ``{"number", "unit", "reason"}`` with reason ``not_in_sources`` (the figure appears in
    neither the sources nor the question) or ``unit_mismatch`` (the sources give that number only with a
    different time unit, for example 7 days versus "7 weeks"). Numbers the question itself contained may be
    restated. Counts only: no text from the answer or sources is returned.
    """
    issues = [{"number": n, "unit": None, "reason": "not_in_sources"}
              for n in ungrounded_numbers(answer, source_text, question)]
    question_numbers = extract_numbers(question)
    source_pairs = _unit_pairs(source_text)
    numbers_with_a_unit = {number for number, _ in source_pairs}
    for number, unit in sorted(_unit_pairs(answer)):
        if number in question_numbers or (number, unit) in source_pairs:
            continue
        if number in numbers_with_a_unit:
            issues.append({"number": number, "unit": unit, "reason": "unit_mismatch"})
    return issues
