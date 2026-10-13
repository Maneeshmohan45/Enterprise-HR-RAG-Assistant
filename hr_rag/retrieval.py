"""Query processing, authorization, hybrid retrieval, reranking, context building."""
from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

from langchain_core.documents import Document

from .cleaning import normalize_for_hash
from .config import ROLE_ACCESS, Settings, settings as default_settings
from .enrichment import STOP_WORDS
from .vectorstore import load_all_chunks

# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------
HR_TERM_MAP = {
    "wfh": "work from home",
    "pto": "paid time off",
    "hr": "human resources",
    "jd": "job description",
}
_TERM_RE = re.compile(r"\b(" + "|".join(map(re.escape, HR_TERM_MAP)) + r")\b", re.IGNORECASE)


def process_query(query: str) -> str:
    """Lower-case the query and ADD the long form of known abbreviations.

    "wfh policy" -> "wfh (work from home) policy". The abbreviation is kept
    because documents contain it too ("HR Manager", "WFH"); replacing it would
    make the keyword search miss those exact words.
    """
    text = " ".join(query.lower().split())
    return _TERM_RE.sub(lambda m: f"{m.group(1).lower()} ({HR_TERM_MAP[m.group(1).lower()]})", text)


def _stem(word: str) -> str:
    """Very light plural stripping so "holidays" matches "holiday"."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_SUBQUESTION_SPLIT = re.compile(
    r"\s*[?;]\s*"
    r"|\s*,\s*(?=(?i:what|how|why|when|who|which|where|is|are|can|does|do|should)\b)"
    r"|(?<=[a-z0-9)])\.\s+(?=[A-Z])"
)


def split_subqueries(question: str, max_parts: int = 4) -> list[str]:
    """Split a multi-part question into its parts ("how many ..., how to apply ...").

    Each part is searched separately so one part cannot crowd out the others.
    Returns an empty list when the question has a single part.
    """
    parts = [p.strip() for p in _SUBQUESTION_SPLIT.split(question) if p and len(p.split()) >= 4]
    return parts[:max_parts] if len(parts) > 1 else []


def tokenize(text: str) -> list[str]:
    """Tokenizer for BM25: lower-case words and numbers, punctuation ignored.

    (LangChain's default just splits on spaces, so "Leave?" would not match
    "leave". The same function is used for the documents and the query.)
    """
    return [_stem(w) for w in re.findall(r"\w+", text.lower())]


# ---------------------------------------------------------------------------
# Authorization (applied BEFORE retrieval)
# ---------------------------------------------------------------------------
def allowed_levels_for(role: str) -> list[str]:
    levels = ROLE_ACCESS.get(role, [])
    if not levels:
        raise PermissionError(f"No document access is configured for role: {role!r}")
    return levels


def build_chroma_filter(levels: list[str], include_archived: bool = False) -> dict:
    access = {"access_level": {"$in": levels}}
    if include_archived:
        return access
    return {"$and": [access, {"is_current": {"$eq": True}}]}


def is_authorized(metadata: dict, levels: list[str], include_archived: bool = False) -> bool:
    """Python mirror of ``build_chroma_filter`` (used for the BM25 corpus)."""
    if metadata.get("access_level", "confidential") not in levels:
        return False
    return include_archived or metadata.get("is_current") is True


# ---------------------------------------------------------------------------
# Fusion + reranking
# ---------------------------------------------------------------------------
def _copy(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=dict(doc.metadata))


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], rrf_k: int = 60,
                           max_results: int = 10) -> list[Document]:
    """Merge ranked lists. score = sum over lists of 1 / (rrf_k + rank)."""
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata.get("chunk_id") or hashlib.sha256(
                doc.page_content.encode("utf-8")).hexdigest()[:24]
            entry = fused.setdefault(key, {"doc": _copy(doc), "score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank)
    ordered = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:max_results]
    results = []
    for entry in ordered:
        entry["doc"].metadata["rrf_score"] = float(entry["score"])
        results.append(entry["doc"])
    return results


class Reranker:
    """Cross-encoder reranker, loaded once on first use.

    If the model cannot be loaded (no internet on first run, too little RAM)
    it falls back to a weak text-similarity score and says so.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.mode = "not_loaded"

    def score(self, query: str, docs: list[Document]) -> list[float]:
        if not docs:
            return []
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            self.mode = "cross_encoder"
            return [float(s) for s in self._model.predict([(query, d.page_content) for d in docs])]
        except Exception as error:
            if self.mode != "fallback":
                print(f"WARNING: reranker unavailable ({error}). Using the weak fallback scorer.")
            self.mode = "fallback"
            return [SequenceMatcher(None, query.lower(), d.page_content.lower()).ratio() for d in docs]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
class BM25Index:
    """Keyword (lexical) search over a fixed list of chunks."""

    def __init__(self, docs: list[Document], k: int):
        from rank_bm25 import BM25Okapi

        self.docs, self.k = docs, k
        self.index = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search(self, query: str) -> list[Document]:
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.docs[i] for i in ranked if scores[i] > 0]  # ignore zero-match chunks


class HybridRetriever:
    def __init__(self, vector_store, reranker, cfg: Settings = default_settings,
                 chunks: list[Document] | None = None):
        self.vector_store = vector_store
        self.reranker = reranker
        self.cfg = cfg
        self.all_chunks = chunks if chunks is not None else load_all_chunks(vector_store)
        self._bm25_cache: dict[tuple, object] = {}

    def _bm25(self, levels: list[str], include_archived: bool):
        key = (tuple(sorted(levels)), include_archived)
        if key not in self._bm25_cache:
            corpus = [c for c in self.all_chunks if is_authorized(c.metadata, levels, include_archived)]
            self._bm25_cache[key] = BM25Index(corpus, self.cfg.top_k) if corpus else None
        return self._bm25_cache[key]

    def _search_one(self, query: str, levels: list[str], include_archived: bool):
        # semantic search, with the permission filter applied inside Chroma
        pairs = self.vector_store.similarity_search_with_score(
            query, k=self.cfg.top_k, filter=build_chroma_filter(levels, include_archived)
        )
        semantic = []
        for doc, distance in pairs:
            doc.metadata["semantic_distance"] = float(distance)
            semantic.append(doc)
        # keyword search over the same authorized corpus
        bm25 = self._bm25(levels, include_archived)
        return semantic, (bm25.search(query) if bm25 else [])

    def retrieve(self, retrieval_query: str, levels: list[str], include_archived: bool = False,
                 sub_queries: list[str] | None = None) -> dict:
        queries = [retrieval_query] + [q for q in (sub_queries or []) if q != retrieval_query]
        ranked_lists, semantic, keyword = [], [], []
        for q in queries:
            sem, kw = self._search_one(q, levels, include_archived)
            ranked_lists += [sem, kw]
            semantic += sem
            keyword += kw

        # fuse everything, then rerank against the full question and each part
        fused = reciprocal_rank_fusion(ranked_lists, self.cfg.rrf_k, self.cfg.rrf_max_results)
        best = [float("-inf")] * len(fused)
        for q in queries:
            for i, score in enumerate(self.reranker.score(q, fused)):
                best[i] = max(best[i], score)
        for doc, score in zip(fused, best):
            doc.metadata["rerank_score"] = score
            doc.metadata["reranker_used"] = getattr(self.reranker, "mode", "custom")
        reranked = sorted(fused, key=lambda d: d.metadata["rerank_score"], reverse=True)
        return {
            "semantic": semantic,
            "keyword": keyword,
            "fused": fused,
            "reranked": reranked[: self.cfg.rerank_top_n],
            "queries": queries,
        }


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
def _query_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[A-Za-z]{3,}\b", text.lower()) if w not in STOP_WORDS}


def compress_chunk_for_query(query: str, text: str, max_sentences: int = 6, max_chars: int = 1400) -> str:
    """Keep the sentences that share the most words with the query."""
    if len(text) <= max_chars:
        return text  # chunks are small; cutting bullet lists drops required skills, steps, etc.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text[:max_chars]
    terms = _query_terms(query)
    scored = [(len(terms & _query_terms(s)), i, s) for i, s in enumerate(sentences)]
    if any(score > 0 for score, _, _ in scored):
        best = sorted(scored, key=lambda t: (t[0], -t[1]), reverse=True)[:max_sentences]
        text = " ".join(s for _, _, s in sorted(best, key=lambda t: t[1]))
    else:
        text = " ".join(sentences[:max_sentences])
    return text[:max_chars]


def describe_location(meta: dict) -> str:
    if meta.get("page_number"):
        return f"Page {meta['page_number']}"
    if meta.get("sheet_name"):
        return f"Sheet {meta['sheet_name']}"
    if meta.get("row_number"):
        return f"Row {meta['row_number']}"
    return ""


def build_context(query: str, docs: list[Document], cfg: Settings = default_settings):
    """Return ``(context_text, sources, selected_docs)``.

    Sources are added one at a time until the character budget is used up, so
    the citation list never contains a source the model did not actually see.
    """
    unique, seen = [], set()
    for doc in docs[: cfg.context_chunks]:
        key = normalize_for_hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    parts, sources, selected = [], [], []
    used = 0
    for doc in unique:
        meta = doc.metadata
        text = doc.page_content
        header = meta.get("chunk_header")
        if header and text.startswith(header):
            text = text[len(header):].lstrip("\n")
        if meta.get("content_kind") == "prose":
            text = compress_chunk_for_query(query, text)

        number = len(parts) + 1
        status = "" if meta.get("is_current", True) else "Status: ARCHIVED (superseded version)\n"
        block = (
            f"SOURCE {number}\nDocument: {meta.get('title') or meta.get('file_name')}\n"
            f"File: {meta.get('file_name')}\nSection: {meta.get('section_title')}\n"
            f"Location: {describe_location(meta) or 'n/a'}\n{status}\n{text}"
        )
        if parts and used + len(block) > cfg.max_context_chars:
            break
        block = block[: cfg.max_context_chars]
        parts.append(block)
        used += len(block) + 2
        selected.append(doc)
        sources.append({
            "chunk_id": meta.get("chunk_id"),
            "document_id": meta.get("document_id"),
            "title": meta.get("title"),
            "file_name": meta.get("file_name"),
            "section_title": meta.get("section_title"),
            "location": describe_location(meta),
            "page_number": meta.get("page_number"),
            "is_current": meta.get("is_current", True),
            "rrf_score": meta.get("rrf_score"),
            "rerank_score": meta.get("rerank_score"),
            "text": text,
        })
    return "\n\n".join(parts), sources, selected