"""Display helpers for streamed answers (text only; nothing here can see roles, levels or documents).

While an answer streams, the chat shows what has arrived so far. Two things must not appear on screen:
citation markers such as ``[S1]`` (the final answer hides them too) and a half-written marker such as
``[S`` or ``[S1``. ``visible_text`` is recomputed from the whole raw text each time, so the result of the
last call equals ``strip_citations(final_text)`` apart from the typing cursor added by the UI.
"""
from __future__ import annotations

import re

from . import citations
from .evaluation import _WORD_NUMBERS
from .generation import StreamAborted, strip_citations

def visible_text(raw: str) -> str:
    """The raw streamed text without complete or half-written citation markers."""
    text = strip_citations(raw)
    return citations.PARTIAL_TAIL.sub("", text)


class ThinkFilter:
    """Remove ``<think>...</think>`` reasoning from a stream, even when a tag is split across pieces."""

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._hold = ""

    def feed(self, piece: str) -> str:
        data, shown = self._hold + piece, []
        self._hold = ""
        while data:
            tag = self._CLOSE if self._inside else self._OPEN
            at = data.find(tag)
            if at >= 0:
                if not self._inside:
                    shown.append(data[:at])
                data, self._inside = data[at + len(tag):], not self._inside
                continue
            keep = next((n for n in range(min(len(tag) - 1, len(data)), 0, -1) if tag.startswith(data[-n:])), 0)
            if not self._inside:
                shown.append(data[:len(data) - keep])
            self._hold = data[len(data) - keep:] if keep else ""
            break
        return "".join(shown)

    def flush(self) -> str:
        """Text held back as a possible tag start that never became one."""
        held, self._hold = self._hold, ""
        return "" if self._inside else held


_NUMBER_TOKEN = re.compile(r"\d|\b(?:" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\b", re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")
_SENTENCE_START = re.compile(r"(?:[.!?]\s+|\n)")
_LIST_NUMBER = re.compile(r"(?m)^[ \t]*\d+[.)](?=\s)")      # "1. Apply ..." numbering is neither a number nor a sentence end


class NumericStreamGuard:
    """Releases streamed text to the screen only after any number in it has been checked.

    Text without numbers is released immediately, so ordinary answers stream word by word. A sentence that
    contains a number is held until it is complete, checked with ``check(text) -> issues``, and only then
    released. If the check finds a problem and ``enforce`` is on, ``StreamAborted`` is raised: the unchecked
    sentence was never shown, and the caller discards the answer. Nothing is shown before the checks.
    """

    def __init__(self, on_delta, check, enforce: bool = True):
        self._on_delta, self._check, self._enforce = on_delta, check, enforce
        self.raw, self._released, self.issues = "", 0, []

    def push(self, piece: str) -> None:
        self.raw += piece
        self._release(final=False)

    def finish(self) -> None:
        self._release(final=True)

    def _release(self, final: bool) -> None:
        while True:
            pending = self.raw[self._released:]
            if not pending:
                return
            blanked = citations.blank_markers(self.raw)[self._released:]
            scan = _LIST_NUMBER.sub(lambda m: " " * len(m.group()), blanked)
            # (a marker's digits and a list's numbering are not numbers to check)
            hold = 0
            if not final:
                partial = citations.PARTIAL_TAIL.search(self.raw)
                if partial and partial.start() >= self._released:
                    hold = len(self.raw) - partial.start()

            risky = _NUMBER_TOKEN.search(scan)
            if risky is None:
                emit_len = len(pending) - hold
                if emit_len > 0:
                    self._emit(emit_len)
                return
            starts = [m.end() for m in _SENTENCE_START.finditer(scan, 0, risky.start())]
            sentence_start = starts[-1] if starts else 0
            if sentence_start:
                self._emit(sentence_start)
                continue
            end = _SENTENCE_END.search(scan, risky.start())
            if end is None and not final:
                if risky.start():
                    self._emit(min(risky.start(), len(pending) - hold))
                return
            sentence = pending[:end.end() if end else len(pending)]
            prefix_match = re.search(r"\S+$", self.raw[:self._released])
            prefix = prefix_match.group() if prefix_match else ""
            clean_sentence = prefix + citations.blank_markers(self.raw[:self._released + (end.end() if end else len(pending))])[self._released:]
            found = self._check(clean_sentence)
            if found:
                self.issues.extend(found)
                if self._enforce:
                    raise StreamAborted(self.issues)
            self._emit(len(sentence))

    def _emit(self, length: int) -> None:
        text = self.raw[self._released:self._released + length]
        self._released += length
        if text:
            self._on_delta(text)
