"""Embedding model + persistent Chroma vector store."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from .config import settings


def get_embeddings(model_name: str | None = None):
    """Load the sentence-transformer embedding model (downloaded on first use)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name or settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # unit vectors => dot product = cosine
    )


def sanitize_metadata(metadata: dict) -> dict:
    """Chroma only stores str/int/float/bool. Convert or drop everything else."""
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            clean[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def open_vector_store(embeddings, persist_dir: Path | None = None,
                      collection_name: str | None = None):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name or settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir or settings.chroma_dir),
    )


def build_vector_store(chunks: list[Document], embeddings, persist_dir: Path | None = None,
                       collection_name: str | None = None, reset: bool = True):
    """Create the index. ``reset=True`` wipes the old collection first so that
    deleted or changed files can never leave stale chunks behind."""
    if reset:
        open_vector_store(embeddings, persist_dir, collection_name).delete_collection()
    store = open_vector_store(embeddings, persist_dir, collection_name)
    prepared = [
        Document(page_content=c.page_content, metadata=sanitize_metadata(c.metadata))
        for c in chunks
    ]
    store.add_documents(documents=prepared, ids=[c.metadata["chunk_id"] for c in chunks])
    return store


def load_all_chunks(store) -> list[Document]:
    """Read every chunk back from Chroma (needed to build the BM25 index)."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
