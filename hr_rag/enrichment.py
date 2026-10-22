"""Rule-based enrichment: summary, keywords and tags (no LLM calls)."""
from __future__ import annotations

import re
from collections import Counter

STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "this", "that", "employee", "employees", "company", "will",
    "must", "should", "all", "any", "may", "from", "their", "they", "have", "has",
}

TAG_RULES = {
    "leave": "leave_management",
    "attendance": "attendance",
    "performance": "performance",
    "security": "information_security",
    "conduct": "code_of_conduct",
    "remote": "remote_work",
    "work from home": "remote_work",
    "onboarding": "onboarding",
    "holiday": "holidays",
    "salary": "compensation",
    "benefit": "benefits",
    "grievance": "grievance",
    "probation": "probation",
}


def create_summary(text: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if len(s.strip()) > 20]
    return " ".join(sentences[:max_sentences])[:300]


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())
    counts = Counter(w for w in words if w not in STOP_WORDS)
    return [w for w, _ in counts.most_common(max_keywords)]


def create_tags(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({tag for keyword, tag in TAG_RULES.items() if keyword in lowered})
