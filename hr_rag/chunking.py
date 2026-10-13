"""Split section-level Documents into retrieval-sized chunks."""
from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def make_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_header(metadata: dict) -> str:
    """One-line label put at the top of every chunk.

    With many files, a chunk such as "Annual Leave: 20 Days" is ambiguous on its
    own. The header ("Leave Policy v2.0 | Leave Entitlements") travels with the
    chunk into the embedding AND the keyword index, which improves retrieval.
    """
    parts = [metadata.get("title"), metadata.get("section_title")]
    return "[" + " | ".join(p for p in parts if p) + "]"


def chunk_documents(docs: list[Document], splitter=None) -> list[Document]:
    splitter = splitter or make_splitter()
    chunks: list[Document] = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        header = chunk_header(doc.metadata)
        for index, piece in enumerate(pieces, start=1):
            chunk_id = hashlib.sha256(
                f"{doc.metadata['document_id']}|{index}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append(
                Document(
                    page_content=f"{header}\n{piece}",
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": index,
                        "total_chunks": len(pieces),
                        "chunk_header": header,
                        "chunk_character_count": len(piece),
                    },
                )
            )
    return chunks
