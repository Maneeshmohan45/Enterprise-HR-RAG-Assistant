"""Text cleaning, validation and exact-duplicate removal."""
from __future__ import annotations

import hashlib
import re
import unicodedata

from langchain_core.documents import Document

# Document types whose records are useless without an ID.
REQUIRED_FIELDS_BY_TYPE: dict[str, list[str]] = {
    "employee_profile": ["employee_id"],
    "employee_directory": ["employee_id"],
}
MIN_CONTENT_CHARS = 15


def clean_text(text: str, unwrap: bool = False) -> str:
    """Normalise extracted text without destroying meaning.

    ``unwrap=True`` re-joins lines that a PDF/TXT broke in the middle of a
    sentence. Only use it for prose: it would damage "key: value" records.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)        # fancy dashes -> "-"
    text = re.sub(r"[\u2022\u25aa\u25e6\u25cf\u2023\u2219]", "-", text)  # bullets -> "-"
    # Drop control / zero-width / private-use characters, keep newlines.
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    # Re-join words hyphenated across a line break: "perfor-\nmance".
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    if unwrap:
        text = re.sub(r"(?<![.!?:;\n])\n(?=[a-z(])", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    cleaned = []
    for doc in docs:
        unwrap = doc.metadata.get("content_kind") == "prose"
        cleaned.append(
            Document(page_content=clean_text(doc.page_content, unwrap=unwrap),
                     metadata=dict(doc.metadata))
        )
    return cleaned


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_document(doc: Document) -> list[str]:
    """Return a list of problems. An empty list means the document is valid."""
    errors: list[str] = []
    text = doc.page_content.strip()
    if not text:
        errors.append("empty_content")
    elif len(text) < MIN_CONTENT_CHARS:
        errors.append("content_too_short")
    for field in REQUIRED_FIELDS_BY_TYPE.get(doc.metadata.get("document_type", ""), []):
        if not str(doc.metadata.get(field, "")).strip():
            errors.append(f"missing_{field}")
    return errors


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def deduplicate(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """Drop exact-text duplicates, keeping the FIRST occurrence.

    The caller sorts so that current documents come first. That way, if an
    archived policy and the current one contain an identical section, the
    archived copy is the one that gets dropped.
    """
    seen: set[str] = set()
    unique, duplicates = [], []
    for doc in docs:
        digest = content_hash(doc.page_content)
        doc.metadata["content_hash"] = digest
        if digest in seen:
            duplicates.append(doc)
        else:
            seen.add(digest)
            unique.append(doc)
    return unique, duplicates
