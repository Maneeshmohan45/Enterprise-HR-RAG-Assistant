"""Post-generation safety checks."""
from __future__ import annotations

import re

from langchain_core.documents import Document

from .retrieval import is_authorized


def verify_access(selected_docs: list[Document], levels: list[str], include_archived: bool) -> bool:
    """Defence in depth: confirm nothing unauthorized reached the prompt."""
    return all(is_authorized(d.metadata, levels, include_archived) for d in selected_docs)


def check_citations(answer: str, source_count: int) -> tuple[bool, list[str], list[str]]:
    """Return ``(all_valid, invalid_ids, used_ids)`` for [S1]-style markers."""
    used = sorted(set(re.findall(r"\[S(\d+)\]", answer)), key=int)
    valid = {str(i) for i in range(1, source_count + 1)}
    invalid = [f"S{n}" for n in used if n not in valid]
    return (not invalid), invalid, [f"S{n}" for n in used]
