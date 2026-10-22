"""The query-time pipeline: question in, grounded and cited answer out."""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Callable

from . import evaluation, guardrails
from .config import Settings, settings as default_settings
from .generation import (build_system_prompt, build_user_prompt, estimate_tokens,
                         generate_answer_with_usage, strip_citations)
from .grounding import question_echo_anomaly
from .request_guard import INJECTION_MESSAGE, RequestAnalysis, analyze_request
from .scope import RESTRICTED_MESSAGE, RESTRICTED_NOTICE, check_scope
from .retrieval import (HybridRetriever, Reranker, allowed_levels_for, build_context,
                        choose_context_budget, process_query, split_subqueries)
from .vectorstore import get_embeddings, open_vector_store

GUARD_MODES = ("off", "shadow", "enforce")

NO_ANSWER = ("I could not find anything in the HR documents you are allowed to access "
             "that answers this question.")


class RAGPipeline:
    def __init__(self, cfg: Settings = default_settings, embeddings=None, vector_store=None,
                 reranker=None, llm: Callable[[str, str], object] | None = None):
        """All heavy parts can be injected, which keeps the tests fast and offline.

        ``llm(system, user)`` may return the answer text, or ``(text, usage_dict)``.
        """
        if cfg.injection_guard_mode not in GUARD_MODES:
            raise ValueError(f"Unsupported INJECTION_GUARD_MODE='{cfg.injection_guard_mode}'. "
                             f"Use one of: {', '.join(GUARD_MODES)}.")
        self.cfg = cfg
        self.embeddings = embeddings or get_embeddings(cfg.embedding_model)
        self.vector_store = vector_store or open_vector_store(
            self.embeddings, cfg.chroma_dir, cfg.collection_name)
        if self.vector_store._collection.count() == 0:
            raise RuntimeError("The vector store is empty. Build it first with:  python ingest.py")
        self.retriever = HybridRetriever(
            self.vector_store, reranker or Reranker(cfg.reranker_model), cfg)
        self.llm = llm or (lambda system, user: generate_answer_with_usage(system, user, cfg))

    def ask(self, question: str, user_context: dict | None = None,
            include_archived: bool = False, log: bool = True, style: str | None = None) -> dict:
        started = time.perf_counter()
        user_context = user_context or {"user_id": "anonymous", "role": "employee"}
        role = user_context.get("role", "employee")
        style = (style or self.cfg.answer_style or "concise").lower()

        # 1. validate the query
        cleaned = " ".join(question.strip().split())
        if not cleaned:
            raise ValueError("Please enter a question.")
        if len(cleaned) > self.cfg.max_query_chars:
            raise ValueError(f"Question is too long (max {self.cfg.max_query_chars} characters).")
        query_id = str(uuid.uuid4())
        # Looks only at the question text (never at roles or documents); acted on after the scope check.
        guard = analyze_request(cleaned) if self.cfg.injection_guard_mode != "off" else RequestAnalysis(False)

        # 2. authorize, then check scope BEFORE any retrieval or LLM call
        levels = allowed_levels_for(role)
        scope = check_scope(cleaned, levels)
        if scope.restricted and not scope.permitted_question:
            response = {"query_id": query_id, "question": cleaned, "retrieval_query": "", "role": role,
                        "style": style, "time": datetime.now().isoformat(timespec="seconds"),
                        "answer": RESTRICTED_MESSAGE, "answer_display": RESTRICTED_MESSAGE, "sources": [],
                        "cited_sources": [], "status": "access_restricted", "evaluation": None,
                        "citations_valid": True, "invalid_citation_ids": [], "cited_ids": [], "usage": {},
                        "timings": {}, "access_notice": None, "scope_rules": list(scope.matched_rules)}
            if log:
                evaluation.log_query(response, self.cfg)
            return response
        # 2a. Clearly detected injection / role-override instructions: decline the WHOLE request, before
        #     any retrieval or LLM call. (A fully restricted request was already declined above, so the
        #     Phase 2 behaviour is unchanged.)
        if guard.declined and self.cfg.injection_guard_mode == "enforce":
            response = {"query_id": query_id, "question": cleaned, "retrieval_query": "", "role": role,
                        "style": style, "time": datetime.now().isoformat(timespec="seconds"),
                        "answer": INJECTION_MESSAGE, "answer_display": INJECTION_MESSAGE, "sources": [],
                        "cited_sources": [], "status": "declined_injection", "evaluation": None,
                        "citations_valid": True, "invalid_citation_ids": [], "cited_ids": [], "usage": {},
                        "timings": {}, "access_notice": None, "scope_rules": list(scope.matched_rules),
                        "injection_rules": list(guard.rules), "request_signals": list(guard.signals),
                        "injection_action": "declined"}
            if log:
                evaluation.log_query(response, self.cfg)
            return response
        # A mixed question continues with only its permitted part; the notice is shown separately.
        effective_question = scope.permitted_question or cleaned

        # 2b. process the query
        retrieval_query = process_query(effective_question)
        sub_queries = [process_query(part) for part in split_subqueries(effective_question)]

        # 3. retrieve (hybrid search + rerank), restricted to what this role may see
        results = self.retriever.retrieve(retrieval_query, levels, include_archived, sub_queries)
        max_chunks, max_chars = choose_context_budget(effective_question, sub_queries, self.cfg)
        context, sources, selected = build_context(
            retrieval_query, results["reranked"], self.cfg, max_chunks, max_chars)
        retrieval_s = time.perf_counter() - started

        base = {"query_id": query_id, "question": cleaned, "retrieval_query": retrieval_query,
                "role": role, "style": style, "time": datetime.now().isoformat(timespec="seconds"),
                "access_notice": RESTRICTED_NOTICE if scope.restricted else None,
                "scope_rules": list(scope.matched_rules),
                "injection_rules": list(guard.rules), "request_signals": list(guard.signals),
                "injection_action": "shadow_would_decline" if guard.declined else None}

        if not selected:  # never call the LLM with an empty context
            response = {**base, "answer": NO_ANSWER, "answer_display": NO_ANSWER, "sources": [],
                        "cited_sources": [], "status": "no_context", "evaluation": None,
                        "citations_valid": True, "invalid_citation_ids": [], "cited_ids": [],
                        "usage": {}, "timings": {"retrieval_s": round(retrieval_s, 3)}}
            if log:
                evaluation.log_query(response, self.cfg)
            return response

        # 4. generate
        system_prompt = build_system_prompt(style)
        user_prompt = build_user_prompt(effective_question, context)
        gen_started = time.perf_counter()
        raw = self.llm(system_prompt, user_prompt)
        generation_s = time.perf_counter() - gen_started
        answer, usage = raw if isinstance(raw, tuple) else (raw, {})
        answer = answer.strip()
        usage = {
            "input_tokens": usage.get("input_tokens") or estimate_tokens(system_prompt + user_prompt),
            "output_tokens": usage.get("output_tokens") or estimate_tokens(answer),
            "estimated": not (usage.get("input_tokens") and usage.get("output_tokens")),
            "context_chunks": len(sources),
        }

        # 5. guardrails
        access_ok = guardrails.verify_access(selected, levels, include_archived)
        citations_valid, invalid_ids, used_ids = guardrails.check_citations(answer, len(sources))
        if not access_ok:
            answer, sources = "You do not have permission to access this information.", []
        status = "access_verified" if access_ok else "access_blocked"

        # Secondary anomaly telemetry only: recorded, never used to change or withhold the answer.
        echo_anomaly = question_echo_anomaly(effective_question, answer, [s["text"] for s in sources])

        # 6. evaluate
        report = evaluation.evaluate(
            question_embedding=self.embeddings.embed_query(retrieval_query),
            answer=answer, context=context, source_texts=[s["text"] for s in sources],
            sources=sources, status=status, citations_valid=citations_valid,
            embeddings=self.embeddings, cfg=self.cfg, question=effective_question)

        # Sources to show as cards: the ones the answer actually cites (else the top 3).
        cited = [s for i, s in enumerate(sources, start=1) if f"S{i}" in used_ids]
        response = {**base, "answer": answer, "answer_display": strip_citations(answer),
                    "sources": sources, "cited_sources": cited or sources[:3], "status": status,
                    "citations_valid": citations_valid, "invalid_citation_ids": invalid_ids,
                    "cited_ids": used_ids, "evaluation": report, "usage": usage, "echo_anomaly": echo_anomaly,
                    "timings": {"retrieval_s": round(retrieval_s, 3),
                                "generation_s": round(generation_s, 3),
                                "total_s": round(time.perf_counter() - started, 3)},
                    "reranker_mode": getattr(self.retriever.reranker, "mode", "custom")}
        if log:
            evaluation.log_query(response, self.cfg)
        return response


def format_response(response: dict, show_all_sources: bool = False) -> str:
    """Human-readable answer followed by its sources (cited ones unless told otherwise)."""
    lines = [response["answer"]]
    if response.get("access_notice"):
        lines += ["", response["access_notice"]]
    sources = response["sources"] if show_all_sources else response.get("cited_sources", response["sources"])
    if sources:
        lines += ["", "Sources", "-------"]
        for s in sources:
            where = f" | {s['location']}" if s["location"] else ""
            old = "" if s["is_current"] else "  (ARCHIVED)"
            lines.append(f"{s['title']} ({s['file_name']}) | {s['section_title']}{where}{old}")
    return "\n".join(lines)
