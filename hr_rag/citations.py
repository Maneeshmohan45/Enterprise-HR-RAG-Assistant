"""One parser for citation markers, shared by display, validation, evaluation and source cards.

Models write markers in several forms: ``[S1]``, ``[S1][S2]``, ``[S1, S2]``, ``[S3,S4]``, ``[S1-S3]``,
``[S2\u2013S6]``. Every part of the system must read them the same way: if the display, the validator and
the source-card code disagreed, a correct answer could leak raw markers or lose its source cards.

Pure functions only: nothing here can see roles, access levels or documents.
"""
from __future__ import annotations

import re

_SEP = r"(?:,|;|-|\u2013|\u2014|to)"
_BODY = rf"\[\s*S\s*\d+(?:\s*{_SEP}\s*S?\s*\d+)*\s*\]"
MARKER = re.compile(_BODY, re.I)
CHAIN = re.compile(rf"\s*{_BODY}(?:\s*,?\s*{_BODY})*", re.I)         # one or more markers in a row
PARTIAL_TAIL = re.compile(r"\s*\[\s*(?:S[\sS\d,;\-\u2013\u2014to]*)?$", re.I)   # a marker still being written
_RANGE_SEP = re.compile(r"-|\u2013|\u2014|\bto\b", re.I)
_MAX_RANGE = 50


def marker_numbers(marker: str) -> list[int]:
    """Numbers one marker refers to; a range such as ``[S1-S3]`` expands to 1, 2, 3."""
    inner = marker.strip()[1:-1]
    numbers: list[int] = []
    previous_end = 0
    for match in re.finditer(r"\d+", inner):
        value = int(match.group())
        between = inner[previous_end:match.start()]
        if numbers and _RANGE_SEP.search(between) and 0 < value - numbers[-1] <= _MAX_RANGE:
            numbers.extend(range(numbers[-1] + 1, value + 1))
        else:
            numbers.append(value)
        previous_end = match.end()
    return numbers


def cited_ids(answer: str) -> list[str]:
    """Sorted unique source ids the answer cites, e.g. ``["S1", "S3"]``."""
    found = {n for marker in MARKER.findall(answer) for n in marker_numbers(marker)}
    return [f"S{n}" for n in sorted(found)]


def strip_markers(text: str) -> str:
    """The text without any citation marker (complete ones only)."""
    text = CHAIN.sub("", text)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()


def blank_markers(text: str) -> str:
    """Replace each marker by spaces of the same length, so positions in the text do not move."""
    return MARKER.sub(lambda m: " " * len(m.group()), text)
