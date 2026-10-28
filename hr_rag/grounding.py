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
from .evaluation import _GROUNDING_STOP, _WORD_NUMBERS, _normalize

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
# A figure is a number written on its own: "85,000" and "1,20,000" are one figure each, and the digits inside
# an identifier such as APS003 or v2 are not figures. Tokens: figures or words.
_FIGURE = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_TOKEN = re.compile(_FIGURE + r"|[a-z]+")
_FIGURE_IN_TEXT = re.compile(r"(?<![\w.])(" + _FIGURE + r")(?![\w])")
_VERSION = re.compile(r"(?<![A-Za-z0-9])v(\d+(?:\.\d+)?)\b", re.I)          # "v2": a version label is a figure to ground
_LIST_NUMBER = re.compile(r"(?m)^\s*(?:#+\s*)?\d+[.)]\s+")
# A sentence that doubts, denies or corrects a figure is not affirming it ("No, 12 days is not allowed").
_NON_AFFIRMING = re.compile(
    r"\b(?:not|no|never|cannot|incorrect|wrong|mistaken|unverified|unspecified|unclear|exceeds?|exceeding|"
    r"more\s+than|less\s+than|instead|rather|however|but|actually)\b|n't\b", re.I)


def _normalize_figure(text: str) -> str:
    text = text.replace(",", "")
    if "." in text:
        text = text.rstrip("0").rstrip(".")            # 2.50 -> 2.5, 10.0 -> 10
    return str(int(text)) if text.isdigit() else text


def figures(text: str) -> set[str]:
    """Figures stated in a text, normalised ("\u20b985,000" -> "85000"). Words count too ("five" = 5)."""
    text = citations.MARKER.sub(" ", text)
    text = _LIST_NUMBER.sub(" ", text)
    found = {_normalize_figure(m) for m in _FIGURE_IN_TEXT.findall(text)}
    found |= {"v" + _normalize_figure(m) for m in _VERSION.findall(text)}
    found |= {str(_WORD_NUMBERS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in _WORD_NUMBERS}
    return found


def _token_number(token: str) -> str | None:
    if token[0].isdigit():
        return _normalize_figure(token)
    return str(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else None


_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n+")


def scan_unit_claims(text: str, before: int = 8, after: int = 6) -> list[dict]:
    """Every "<number> <time unit>" in a text, with the content words around it.

    ``{"number": "7", "unit": "day", "unit_text": "Days", "words": {...}}``. The words say what the figure is
    about ("paternity", "leave"), so a figure can be compared only with figures about the same thing. The
    words are taken from the same sentence or line only: a neighbouring sentence is about something else.
    """
    claims = []
    for segment in _SEGMENT.split(citations.blank_markers(text)):
        raw = re.findall(_FIGURE + r"|[A-Za-z]+", segment)
        tokens = [t.lower() for t in raw]
        for i, token in enumerate(tokens):
            number = _token_number(token)
            if number is None:
                continue
            for offset, following in enumerate(tokens[i + 1:i + 4], start=1):
                if following in _TIME_UNITS:
                    around = tokens[max(0, i - before):i] + tokens[i + 1 + offset:i + 1 + offset + after]
                    words = {w for w in around if w.isalpha() and len(w) > 2 and w not in _GROUNDING_STOP
                             and w not in _TIME_UNITS and w not in _WORD_NUMBERS and w not in _UNIT_SKIP}
                    claims.append({"number": number, "unit": _TIME_UNITS[following],
                                   "unit_text": raw[i + offset], "words": words})
                    break
                if following not in _UNIT_SKIP:
                    break
    return claims


def _unit_pairs(text: str) -> set[tuple[str, str]]:
    return {(c["number"], c["unit"]) for c in scan_unit_claims(text)}


def _affirmed_anywhere(answer: str, number: str, unit: str | None = None) -> bool:
    """Is the figure stated, in at least one sentence of the answer, without doubt, denial or correction?"""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", answer) if s.strip()]
    relevant = [s for s in sentences
                if number in figures(s) and (unit is None or (number, unit) in _unit_pairs(s))]
    return not relevant or any(not _NON_AFFIRMING.search(s) for s in relevant)


def numeric_issues(answer: str, source_text: str, question: str = "") -> list[dict]:
    """Figures in the answer that the sources do not support.

    Each issue is ``{"number", "unit", "reason"}``:
      * ``not_in_sources``      the figure appears in neither the sources nor the question (invented or calculated);
      * ``claimed_in_question`` the figure is in the QUESTION but not in the sources, and the answer states it as
                                fact. Something the user said is not evidence: "Your HR is correct, the bonus is
                                4 months" is withheld unless the sources say so;
      * ``unit_mismatch``       the sources give that number only with a different time unit (7 days, not 7 weeks).

    A figure from the question may still be restated when the answer doubts, denies or corrects it ("No, 12 days
    is not allowed; the limit is 10 days"). Counts only: no text from the answer or sources is returned.
    """
    issues: list[dict] = []
    source_figures, question_figures = figures(source_text), figures(question)
    for number in sorted(figures(answer) - source_figures, key=lambda n: float(n.lstrip("v"))):
        if number not in question_figures:
            issues.append({"number": number, "unit": None, "reason": "not_in_sources"})
        elif _affirmed_anywhere(answer, number):
            issues.append({"number": number, "unit": None, "reason": "claimed_in_question"})
    source_pairs = _unit_pairs(source_text)
    numbers_with_a_unit = {number for number, _ in source_pairs}
    question_pairs = _unit_pairs(question)
    for number, unit in sorted(_unit_pairs(answer)):
        if (number, unit) in source_pairs or number not in numbers_with_a_unit:
            continue
        if (number, unit) in question_pairs and not _affirmed_anywhere(answer, number, unit):
            continue                                     # the answer is correcting the user's wrong unit
        issues.append({"number": number, "unit": unit, "reason": "unit_mismatch"})
    return issues
