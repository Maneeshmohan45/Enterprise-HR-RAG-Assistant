"""Shared fixtures. Tests run fully offline: no model downloads, no API calls."""
import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import Embeddings  # noqa: E402

from hr_rag.config import Settings  # noqa: E402
from hr_rag.ingestion import prepare_chunks  # noqa: E402
from hr_rag.vectorstore import build_vector_store  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Hashed bag-of-words vectors, L2-normalised. Similar wording => similar vector."""
    DIM = 512

    def _vec(self, text):
        v = [0.0] * self.DIM
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    """Scores by word overlap with the query. Stands in for the cross-encoder."""
    mode = "fake"

    def score(self, query, docs):
        q = set(re.findall(r"\w+", query.lower()))
        return [len(q & set(re.findall(r"\w+", d.page_content.lower()))) / (len(q) or 1) for d in docs]


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("hr")
    return Settings(chroma_dir=root / "chroma", logs_dir=root / "logs")


@pytest.fixture(scope="session")
def prepared(cfg):
    return prepare_chunks(cfg, verbose=False)


@pytest.fixture(scope="session")
def chunks(prepared):
    return prepared[0]


@pytest.fixture(scope="session")
def report(prepared):
    return prepared[1]


@pytest.fixture(scope="session")
def embeddings():
    return FakeEmbeddings()


@pytest.fixture(scope="session")
def store(chunks, embeddings, cfg):
    return build_vector_store(chunks, embeddings, cfg.chroma_dir, cfg.collection_name)


@pytest.fixture(scope="session")
def fake_reranker():
    return FakeReranker()
