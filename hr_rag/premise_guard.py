"""Things the USER says are not evidence.

Two deterministic checks that stop the assistant from agreeing with something only the user claimed:

1. ``is_statement_only``: an input that only asserts something ("My salary is 85,000.", "My HR said the
   bonus is 4 months now.") and asks nothing. There is nothing to answer, and answering would confirm the claim.
2. ``find_premise_conflicts``: a question that states a figure and unit ("Since paternity leave is 7 weeks, ...")
   when the sources give the same number about the same thing with a different unit (7 days). The question
   is built on a wrong premise, so it is not answered; the user is shown the figure the documents give.

Both only ever DECLINE (narrow what the assistant does). They take text and already-retrieved sources and
cannot see roles, access levels or documents beyond what retrieval already allowed.
"""
from __future__ import annotations

import re
import unicodedata

from .grounding import scan_unit_claims

STATEMENT_MESSAGE = ("I can't confirm statements about pay, policy or entitlements. Please ask a question about "
                     "our HR policies and documents, and I'll answer from them.")
PREMISE_PREFIX = "Your question says "

_REQUEST_START = re.compile(
    r"^(?:please\s+)?(?:what|what's|whats|how|why|when|where|who|whom|whose|which|is|are|am|was|were|do|does|did|"
    r"can|could|would|will|shall|should|may|might|list|give|show|tell|explain|describe|compare|find|provide|state|"
    r"name|outline|summari[sz]e|print|display|calculate|compute|write|create|generate|draft|translate|help|check|"
    r"confirm|verify|clarify|define|enumerate|share|send|get|fetch|look|search|open|apply|request|submit)\b", re.I)
_REQUEST_FRAME = re.compile(
    r"\bplease\b|\bi\s+(?:need|want|would\s+like|'d\s+like|wish|am\s+(?:looking|trying|asking)|was\s+wondering)\b|"
    r"\b(?:can|could|would|will)\s+you\b|\bhelp\s+me\b|\btell\s+me\b|\blet\s+me\s+know\b|\bi\s+wonder\b|"
    r"\bi'?m\s+(?:curious|wondering|looking|trying)\b", re.I)
_CLAIM = re.compile(
    r"\b(?:is|are|was|were|will\s+be|has\s+been|have\s+been|should|must|shall|"
    r"have\s+(?:already\s+)?(?:completed|finished|submitted|served))\b|"
    r"\b(?:said|says|told|announced|confirmed|stated|informed|updated|decided|approved|authori[sz]ed|promised|"
    r"mentioned|encouraged|entitled|completed|finished|submitted|served)\b|\b(?:effective|immediately)\b", re.I)
_PLEASANTRY = re.compile(r"\b(?:thank(?:s|\s+you)|thx|appreciate|helpful|great|awesome|perfect|cheers|bye|goodbye|"
                         r"hello|hi|hey)\b", re.I)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSES = re.compile(r"(?<=[.!?])\s+|\n+|[,;:]\s+")
_LEADING_FILLER = re.compile(r"^(?:so|and|also|ok|okay|well|then|now|hey|hi|hello|btw|actually)\b[\s,]*", re.I)
_WH_ANYWHERE = re.compile(r"\b(?:what|how|why)\b", re.I)         # "what is the difference between ..." without a ?
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad"), None)


def is_statement_only(text: str) -> bool:
    """True for an input that asserts something and asks for nothing.

    Keyword-style queries ("casual leave policy"), questions without a question mark, commands ("list the
    holidays") and first-person requests ("I need the leave policy") are NOT statements.
    """
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).strip()
    words = text.split()
    if len(words) < 3 or "?" in text:
        return False
    if len(words) <= 12 and _PLEASANTRY.search(text):
        return False
    if _REQUEST_FRAME.search(text) or _WH_ANYWHERE.search(text):
        return False
    clauses = [_LEADING_FILLER.sub("", c.strip(" \t\"'()[]")) for c in _CLAUSES.split(text) if c.strip()]
    if any(_REQUEST_START.match(c) for c in clauses):          # "Using the record, list ..." / "For each, identify ..."
        return False
    return bool(_CLAIM.search(text))


def find_premise_conflicts(question: str, sources: list[dict]) -> list[dict]:
    """Figures the question states that the sources contradict.

    A question figure ("7 weeks") conflicts when a source gives the SAME number about the SAME thing with a
    different time unit ("7 Days"), and no source supports the question's own unit. "The same thing" means the
    words around the two figures overlap (paternity, leave): an unrelated "7 days" elsewhere is not a conflict.
    Returns ``[{"claimed", "found", "file_name"}]``.
    """
    conflicts, seen = [], set()
    for claim in scan_unit_claims(question):
        supported, contradiction = False, None
        for source in sources:
            text = f'{source.get("section_title") or ""} {source["text"]}'
            for segment in scan_unit_claims(text):
                if segment["number"] != claim["number"] or len(segment["words"] & claim["words"]) < 2:
                    continue
                if segment["unit"] == claim["unit"]:
                    supported = True
                elif contradiction is None:
                    contradiction = (segment, source.get("file_name"))
        if contradiction and not supported:
            segment, file_name = contradiction
            key = (claim["number"], claim["unit"], segment["unit"])
            if key not in seen:
                seen.add(key)
                conflicts.append({"claimed": f'{claim["number"]} {claim["unit_text"].lower()}',
                                  "found": f'{segment["number"]} {segment["unit_text"].lower()}',
                                  "file_name": file_name})
    return conflicts


def premise_message(conflicts: list[dict]) -> str:
    """A fixed-shape correction built only from the user's own figure and the figure in the documents."""
    parts = [f'{c["claimed"]}, but the HR documents ({c["file_name"]}) give {c["found"]} for this'
             for c in conflicts[:2]]
    return f"{PREMISE_PREFIX}{' and '.join(parts)}. Please check the figure and ask again."
