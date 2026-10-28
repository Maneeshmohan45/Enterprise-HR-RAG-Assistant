"""Response classes: what kind of response was this, and was it handled correctly for that kind?

One ordinary QA score cannot judge every response. A refusal is right because the system refused and
attached nothing; a "not in the documents" answer is right because it invented nothing; only a normal
answer is judged on grounding. This module assigns the class and runs the checks that fit it.

Classes
    NORMAL_GROUNDED   an answer from the documents (judged by the existing quality checks)
    SECURITY_REFUSAL  the request guard declined an injection / role-override request
    ACCESS_RESTRICTED the request needs a higher access level (scope check or access verification)
    OUT_OF_DOMAIN     not an HR question (domain rules, relevance gate, or a model decline)
    NOT_FOUND         the documents do not contain it (no context, or the model says so)

Runtime limits (stated honestly): the system can prove it refused and attached nothing; it cannot
prove a refusal was *warranted*. That needs labelled test cases (see scripts/v1_acceptance.py).
This module also decides which source cards may be shown: cards are shown only when the answer is a
normal answer, cites the source, and actually shares content with it. There is no "top 3" fallback.
"""
from __future__ import annotations

import re

from . import citations
from .evaluation import _GROUNDING_STOP, _normalize, ungrounded_numbers

NORMAL, SECURITY, ACCESS, OUT_OF_DOMAIN, NOT_FOUND = (
    "NORMAL_GROUNDED", "SECURITY_REFUSAL", "ACCESS_RESTRICTED", "OUT_OF_DOMAIN", "NOT_FOUND")
CLASSES = (NORMAL, SECURITY, ACCESS, OUT_OF_DOMAIN, NOT_FOUND)

# Status values set by the application itself (the model was not asked, or its answer was discarded).
APP_DECLINE_CLASS = {"declined_injection": SECURITY, "access_restricted": ACCESS, "access_blocked": ACCESS,
                     "declined_out_of_domain": OUT_OF_DOMAIN, "no_context": NOT_FOUND,
                     "declined_ungrounded": NOT_FOUND, "declined_statement": NOT_FOUND,
                     "declined_premise": NOT_FOUND}

_ABSENCE = re.compile(
    r"\b(?:does\s+not|doesn't|do\s+not|don't|did\s+not|is\s+not|isn't|are\s+not|aren't)\s+"
    r"(?:specify|specified|mention|mentioned|say|state|stated|contain|cover|covered|include|included|provide|"
    r"describe|address|list)\b"
    r"|\bno\s+(?:information|mention|details?|reference|specific|policy)\b"
    r"|\b(?:could\s+not|couldn't|cannot|can't|can\s+not|unable\s+to)\s+(?:find|locate|see)\b"
    r"|\bnot\s+(?:found|available)\s+in\s+(?:the|these|this|my)\b|\bnothing\s+(?:in|about)\b", re.I)
_DECLINE = re.compile(
    r"\bi\s+(?:can\s+only|only|can)\s+(?:help|answer|assist)\b[^.?!]{0,40}\b(?:hr|policies|documents|company)\b"
    r"|\bi\s+(?:can't|cannot|can\s+not|am\s+not\s+able\s+to|'m\s+not\s+able\s+to)\s+(?:help|assist)\s+with\b"
    r"|\b(?:outside|beyond)\s+(?:of\s+)?(?:the\s+|my\s+)?(?:scope|hr)\b"
    r"|\bonly\s+(?:answer|help\s+with)\s+questions\s+about\b|\bnot\s+an?\s+hr\s+(?:question|topic)\b", re.I)
_MAX_REFUSAL_WORDS = 80


def detect_model_refusal(answer: str, cited_ids: list[str]) -> str | None:
    """Return "decline" (not an HR question), "absence" (not in the documents) or None.

    Only short answers that cite nothing count: an answer that cites a source is giving grounded
    information, even if it also says one detail is not specified.
    """
    if cited_ids or len(answer.split()) > _MAX_REFUSAL_WORDS:
        return None
    if _DECLINE.search(answer):
        return "decline"
    if _ABSENCE.search(answer):
        return "absence"
    return None


def classify(status: str, answer: str, cited_ids: list[str]) -> str:
    """The behavioural class of one response."""
    if status in APP_DECLINE_CLASS:
        return APP_DECLINE_CLASS[status]
    kind = detect_model_refusal(answer, cited_ids)
    if kind == "decline":
        return OUT_OF_DOMAIN
    if kind == "absence":
        return NOT_FOUND
    return NORMAL


def _words(text: str) -> set[str]:
    text = citations.MARKER.sub(" ", text)
    return {w for w in _normalize(text).split() if w not in _GROUNDING_STOP and len(w) > 2}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def source_supports(answer: str, source_text: str) -> bool:
    """A cited source must share something concrete with the answer: a number, or at least two content
    words (one if the answer has only one). A source that shares nothing cannot be what the answer used."""
    answer_words, shared = _words(answer), _words(answer) & _words(source_text)
    if _numbers(answer) & _numbers(source_text):
        return True
    return len(shared) >= min(2, len(answer_words)) and bool(shared)


def select_source_cards(answer: str, sources: list[dict], cited_ids: list[str], response_class: str) -> tuple[list[dict], list[str]]:
    """Cards to show, and cited ids that were hidden because the source does not support the answer.

    Only a normal answer shows cards, only for sources it cites, only when the source supports it.
    Nothing is shown merely because retrieval happened (there is no "top 3" fallback).
    """
    if response_class != NORMAL:
        return [], []
    cards, unsupported = [], []
    for index, source in enumerate(sources, start=1):
        if f"S{index}" not in cited_ids:
            continue
        if source_supports(answer, source["text"]):
            cards.append(source)
        else:
            unsupported.append(f"S{index}")
    return cards, unsupported


def assess(response_class: str, *, llm_called: bool, answer: str, cards: list[dict],
           context_text: str, question: str, fixed_messages: tuple[str, ...],
           unsupported_ids: list[str], fixed_prefixes: tuple[str, ...] = ()) -> dict:
    """Checks that fit the class. ``passed`` is True/False for refusals and not-found answers, and None
    for normal answers (those are judged by the existing quality score)."""
    if response_class == NORMAL:
        return {"class": NORMAL, "checks": {"citations_supported": not unsupported_ids}, "passed": None}
    checks = {"no_source_cards": not cards}
    if not llm_called:                       # the application declined; nothing came from the model
        checks["llm_not_called"] = True
        checks["fixed_message"] = answer in fixed_messages or answer.startswith(fixed_prefixes)
    else:                                    # the model declined or said "not found"
        checks["no_invented_numbers"] = not ungrounded_numbers(answer, context_text, question)
        if answer in fixed_messages:         # the model's answer was discarded and replaced by the fixed message
            checks["fixed_message"] = True
    return {"class": response_class, "checks": checks, "passed": all(checks.values())}
