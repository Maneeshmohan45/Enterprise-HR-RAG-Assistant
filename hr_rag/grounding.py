"""Answer-side anomaly telemetry.

Step 1 contains only ``question_echo_anomaly``: a SECONDARY signal that notices when an answer's
content words come from the question instead of the sources. It is recorded and shown in analysis
only. It never changes, withholds or rejects an answer: repeating words from the question is
normal, and a single heuristic must not decide what a user sees. The primary defences are the
request guard (decline clear injection), the Phase 1 filter and, later, the grounding gate which
judges answers only by how well the SOURCES support them.
"""
from __future__ import annotations

import re

from .evaluation import _GROUNDING_STOP, _normalize

MIN_QUESTION_ONLY_TERMS = 4     # telemetry thresholds; provisional until calibrated on real answers
MIN_QUESTION_ONLY_RATIO = 0.25


def _content_words(text: str) -> set[str]:
    text = re.sub(r"\[S\d+\]", " ", text)
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
