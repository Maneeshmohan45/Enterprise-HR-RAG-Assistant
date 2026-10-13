"""The query-time pipeline: question in, grounded and cited answer out."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Callable

from . import evaluation, guardrails
from .config import Settings, settings as default_settings
from .generation import SYSTEM_INSTRUCTION, build_user_prompt, generate_answer
from .retrieval import (HybridRetriever, Reranker, allowed_levels_for, build_context, process_query,
                        split_subqueries)
from .vectorstore import get_embeddings, open_vector_store

NO_ANSWER = ("I could not find anything in the HR documents you are allowed to access "
             "that answers this question.")


class RAGPipeline:
    def __init__(self, cfg: Settings = default_settings, embeddings=None, vector_store=None,
                 reranker=None, llm: Callable[[str, str], str] | None = None):
        """All heavy parts can be injected, which keeps the tests fast and offline."""
        self.cfg = cfg
        self.embeddings = embeddings or get_embeddings(cfg.embedding_model)
        self.vector_store = vector_store or open_vector_store(
            self.embeddings, cfg.chroma_dir, cfg.collection_name)
        if self.vector_store._collection.count() == 0:
            raise RuntimeError("The vector store is empty. Build it first with:  python ingest.py")
        self.retriever = HybridRetriever(
            self.vector_store, reranker or Reranker(cfg.reranker_model), cfg)
        self.llm = llm or (lambda system, user: generate_answer(system, user, cfg))

    def ask(self, question: str, user_context: dict | None = None,
            include_archived: bool = False, log: bool = True) -> dict:
        user_context = user_context or {"user_id": "anonymous", "role": "employee"}
        role = user_context.get("role", "employee")

        # 1. validate the query
        cleaned = " ".join(question.strip().split())
        if not cleaned:
            raise ValueError("Please enter a question.")
        if len(cleaned) > self.cfg.max_query_chars:
            raise ValueError(f"Question is too long (max {self.cfg.max_query_chars} characters).")
        query_id = str(uuid.uuid4())

        # 2. authorize + process the query
        levels = allowed_levels_for(role)
        retrieval_query = process_query(cleaned)

        # 3. retrieve (hybrid search + rerank), restricted to what this role may see
        sub_queries = [process_query(part) for part in split_subqueries(cleaned)]
        results = self.retriever.retrieve(retrieval_query, levels, include_archived, sub_queries)
        context, sources, selected = build_context(retrieval_query, results["reranked"], self.cfg)

        base = {"query_id": query_id, "question": cleaned, "retrieval_query": retrieval_query,
                "role": role, "time": datetime.now().isoformat(timespec="seconds")}

        if not selected:  # never call the LLM with an empty context
            response = {**base, "answer": NO_ANSWER, "sources": [], "status": "no_context",
                        "evaluation": None, "citations_valid": True, "invalid_citation_ids": []}
            if log:
                evaluation.log_query(response, self.cfg)
            return response

        # 4. generate
        answer = self.llm(SYSTEM_INSTRUCTION, build_user_prompt(cleaned, context)).strip()

        # 5. guardrails
        access_ok = guardrails.verify_access(selected, levels, include_archived)
        citations_valid, invalid_ids, used_ids = guardrails.check_citations(answer, len(sources))
        if not access_ok:
            answer, sources = "You do not have permission to access this information.", []
        status = "access_verified" if access_ok else "access_blocked"

        # 6. evaluate
        report = evaluation.evaluate(
            question_embedding=self.embeddings.embed_query(retrieval_query),
            answer=answer, context=context, source_texts=[s["text"] for s in sources],
            sources=sources, status=status, citations_valid=citations_valid,
            embeddings=self.embeddings, cfg=self.cfg, question=cleaned)

        response = {**base, "answer": answer, "sources": sources, "status": status,
                    "citations_valid": citations_valid, "invalid_citation_ids": invalid_ids,
                    "cited_ids": used_ids, "evaluation": report,
                    "reranker_mode": getattr(self.retriever.reranker, "mode", "custom")}
        if log:
            evaluation.log_query(response, self.cfg)
        return response


def format_response(response: dict) -> str:
    """Human-readable answer followed by its sources."""
    lines = [response["answer"]]
    if response["sources"]:
        lines += ["", "Sources", "-------"]
        for i, s in enumerate(response["sources"], start=1):
            where = f" | {s['location']}" if s["location"] else ""
            old = "" if s["is_current"] else "  (ARCHIVED)"
            lines.append(f"[S{i}] {s['title']} ({s['file_name']}) | {s['section_title']}{where}{old}")
    return "\n".join(lines)