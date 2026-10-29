"""The query-time pipeline: question in, grounded and cited answer out."""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Callable

from . import assessment, evaluation, guardrails
from .config import Settings, settings as default_settings
from .generation import (StreamAborted, build_system_prompt, build_user_prompt, estimate_tokens,
                         generate_answer_with_usage, strip_citations, stream_answer_with_usage)
from .streaming import NumericStreamGuard
from .domain_guard import DOMAIN_PARTIAL_NOTICE, OUT_OF_DOMAIN_MESSAGE, DomainAnalysis, analyze_domain
from .grounding import numeric_issues, question_echo_anomaly
from .roster import ROSTER_NOTICE, RosterPlan, plan_roster_answer
from .premise_guard import (PREMISE_PREFIX, STATEMENT_MESSAGE, find_premise_conflicts, is_statement_only,
                            premise_message)
from .request_guard import INJECTION_MESSAGE, RequestAnalysis, analyze_request
from .scope import RESTRICTED_MESSAGE, RESTRICTED_NOTICE, check_scope
from .retrieval import (HybridRetriever, Reranker, allowed_levels_for, build_context,
                        choose_context_budget, process_query, split_subqueries)
from .vectorstore import get_embeddings, open_vector_store

GUARD_MODES = ("off", "shadow", "enforce")
ACCESS_BLOCKED_MESSAGE = "You do not have permission to access this information."
INSUFFICIENT_INFO_MESSAGE = ("I couldn't find sufficient information in the provided sources to answer this "
                             "question reliably.")

NO_ANSWER = ("I could not find anything in the HR documents you are allowed to access "
             "that answers this question.")


class RAGPipeline:
    def __init__(self, cfg: Settings = default_settings, embeddings=None, vector_store=None,
                 reranker=None, llm: Callable[[str, str], object] | None = None,
                 llm_stream: Callable[[str, str, Callable[[str], None]], object] | None = None):
        """All heavy parts can be injected, which keeps the tests fast and offline.

        ``llm(system, user)`` may return the answer text, or ``(text, usage_dict)``.
        ``llm_stream(system, user, on_delta)`` is the optional streaming variant; with the default
        provider it is the provider's streaming call. A custom ``llm`` without ``llm_stream`` simply
        does not stream (``ask(on_delta=...)`` then receives the whole answer once).
        """
        for name, value in (("INJECTION_GUARD_MODE", cfg.injection_guard_mode),
                            ("DOMAIN_GUARD_MODE", cfg.domain_guard_mode),
                            ("RELEVANCE_GATE_MODE", cfg.relevance_gate_mode),
                            ("NUMERIC_GATE_MODE", cfg.numeric_gate_mode),
                            ("PREMISE_GUARD_MODE", cfg.premise_guard_mode),
                            ("ROSTER_LOOKUP_MODE", cfg.roster_lookup_mode)):
            if value not in GUARD_MODES:
                raise ValueError(f"Unsupported {name}='{value}'. Use one of: {', '.join(GUARD_MODES)}.")
        if cfg.relevance_gate_mode == "enforce" and cfg.relevance_min_rerank_score is None:
            raise ValueError("RELEVANCE_GATE_MODE=enforce needs RELEVANCE_MIN_RERANK_SCORE. Calibrate it "
                             "with scripts/calibrate_relevance.py first, or keep the mode on 'shadow'.")
        self.cfg = cfg
        self.embeddings = embeddings or get_embeddings(cfg.embedding_model)
        self.vector_store = vector_store or open_vector_store(
            self.embeddings, cfg.chroma_dir, cfg.collection_name)
        if self.vector_store._collection.count() == 0:
            raise RuntimeError("The vector store is empty. Build it first with:  python ingest.py")
        self.retriever = HybridRetriever(
            self.vector_store, reranker or Reranker(cfg.reranker_model), cfg)
        self.llm = llm or (lambda system, user: generate_answer_with_usage(system, user, cfg))
        self.llm_stream = llm_stream or (
            (lambda system, user, on_delta: stream_answer_with_usage(system, user, cfg, on_delta))
            if llm is None else None)
        self._fixed_messages = (RESTRICTED_MESSAGE, INJECTION_MESSAGE, OUT_OF_DOMAIN_MESSAGE, NO_ANSWER,
                                ACCESS_BLOCKED_MESSAGE, INSUFFICIENT_INFO_MESSAGE, STATEMENT_MESSAGE)
        self._fixed_prefixes = (PREMISE_PREFIX,)          # the correction message carries the user's own figure

    # -- helpers ---------------------------------------------------------------------------------
    def _finish_declined(self, response: dict, log: bool, llm_called: bool = False) -> dict:
        """Class and checks for a response the APPLICATION produced (the model was not used, or its answer was
        discarded: ``llm_called=True``), then log."""
        response_class = assessment.APP_DECLINE_CLASS[response["status"]]
        result = assessment.assess(response_class, llm_called=llm_called, answer=response["answer"],
                                   cards=response["cited_sources"], context_text="",
                                   question=response["question"], fixed_messages=self._fixed_messages,
                                   unsupported_ids=[], fixed_prefixes=self._fixed_prefixes)
        response.update(response_class=response_class, class_checks=result["checks"],
                        class_passed=result["passed"])
        if log:
            evaluation.log_query(response, self.cfg)
        return response

    def _declined(self, *, query_id: str, question: str, role: str, style: str, status: str, message: str,
                  scope, guard, **extra) -> dict:
        return {"query_id": query_id, "question": question, "retrieval_query": extra.pop("retrieval_query", ""),
                "role": role, "style": style, "time": datetime.now().isoformat(timespec="seconds"),
                "answer": message, "answer_display": message, "sources": [], "cited_sources": [],
                "status": status, "evaluation": None, "citations_valid": True, "invalid_citation_ids": [],
                "cited_ids": [], "usage": {}, "timings": extra.pop("timings", {}), "access_notice": None,
                "scope_rules": list(scope.matched_rules), "injection_rules": list(guard.rules),
                "request_signals": list(guard.signals), "injection_action": None, **extra}

    def _withheld(self, base: dict, issues: list[dict], mode: str, retrieval_s: float, generation_s: float,
                  input_tokens: int, output_tokens: int, context_chunks: int, log: bool,
                  diagnostic_sources: list[dict] | None = None) -> dict:
        """The generated answer is discarded: only the fixed message is returned, with no sources."""
        return self._finish_declined({
            **base, "answer": INSUFFICIENT_INFO_MESSAGE, "answer_display": INSUFFICIENT_INFO_MESSAGE,
            "sources": [], "cited_sources": [], "status": "declined_ungrounded", "evaluation": None,
            "citations_valid": True, "invalid_citation_ids": [], "cited_ids": [],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "estimated": True,
                      "context_chunks": context_chunks, "answer_discarded": True},
            "numeric_gate": {"mode": mode, "issues": issues},
            "diagnostics": {
                "stage": "numeric_gate",
                "rejection_reason": "unsupported_figures_or_units",
                "source_files": [s["file_name"] for s in (diagnostic_sources or [])],
                "retrieved_chunk_count": len(diagnostic_sources or []),
            },
            "timings": {"retrieval_s": round(retrieval_s, 3), "generation_s": round(generation_s, 3),
                        "total_s": round(retrieval_s + generation_s, 3)}}, log, llm_called=True)

    def _relevance_gate(self, results: dict) -> dict:
        """Compare the best reranker score with the calibrated threshold. Only records unless enforced."""
        scores = [d.metadata.get("rerank_score") for d in results["reranked"]
                  if d.metadata.get("rerank_score") is not None]
        top = max(scores) if scores else None
        threshold = self.cfg.relevance_min_rerank_score
        # A fallback scorer (not the real cross-encoder) has a different scale, so the gate stands down.
        active = (self.cfg.relevance_gate_mode != "off" and threshold is not None and top is not None
                  and getattr(self.retriever.reranker, "mode", "") != "fallback")
        return {"mode": self.cfg.relevance_gate_mode, "active": active,
                "top_score": None if top is None else round(float(top), 4), "threshold": threshold,
                "would_decline": bool(active and top < threshold)}

    def ask(self, question: str, user_context: dict | None = None,
            include_archived: bool = False, log: bool = True, style: str | None = None,
            on_delta: Callable[[str], None] | None = None) -> dict:
        """Answer one question. ``on_delta(text)`` (optional) receives the answer as it is generated.

        Nothing is ever passed to ``on_delta`` before every pre-generation check (scope, request guard,
        domain rules, relevance gate, access verification) has passed.
        """
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
            return self._finish_declined(response, log)
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
            return self._finish_declined(response, log)
        # A mixed question continues with only its permitted part; the notice is shown separately.
        effective_question = scope.permitted_question or cleaned

        # 2a-2. Clearly out-of-domain requests (arithmetic, code, creative writing, general knowledge):
        #       decline before retrieval, so unrelated HR text can never look like a source.
        domain = (analyze_domain(effective_question) if self.cfg.domain_guard_mode != "off"
                  else DomainAnalysis(False))
        if domain.out_of_domain and self.cfg.domain_guard_mode == "enforce":
            return self._finish_declined(self._declined(
                query_id=query_id, question=cleaned, role=role, style=style, status="declined_out_of_domain",
                message=OUT_OF_DOMAIN_MESSAGE, scope=scope, guard=guard, domain_category=domain.category,
                domain_action="declined"), log)
        # A compound request ("limit for casual leave, and also write a bash one-liner"): answer the HR part
        # only, and say that the rest was not answered. The non-HR clause never reaches the model.
        domain_notice = None
        if domain.permitted_question and self.cfg.domain_guard_mode == "enforce":
            domain_notice, effective_question = DOMAIN_PARTIAL_NOTICE, domain.permitted_question

        # 2a-3. A statement that asks nothing ("My salary is 85,000.") has nothing to answer, and answering it
        #       would confirm what only the user claimed. Decline before retrieval.
        statement = self.cfg.premise_guard_mode != "off" and is_statement_only(effective_question)
        if statement and self.cfg.premise_guard_mode == "enforce":
            return self._finish_declined(self._declined(
                query_id=query_id, question=cleaned, role=role, style=style, status="declined_statement",
                message=STATEMENT_MESSAGE, scope=scope, guard=guard, statement_action="declined"), log)

        # 2b. process the query
        retrieval_query = process_query(effective_question)
        sub_queries = [process_query(part) for part in split_subqueries(effective_question)]

        # 3. retrieve (hybrid search + rerank), restricted to what this role may see
        results = self.retriever.retrieve(retrieval_query, levels, include_archived, sub_queries)
        max_chunks, max_chars = choose_context_budget(effective_question, sub_queries, self.cfg)
        context, sources, selected = build_context(
            retrieval_query, results["reranked"], self.cfg, max_chunks, max_chars)
        retrieval_s = time.perf_counter() - started

        # 3a. Relevance gate (isolated second layer; records only until it is calibrated and enforced).
        gate = self._relevance_gate(results)
        if self.cfg.relevance_gate_mode == "enforce" and gate["would_decline"]:
            return self._finish_declined(self._declined(
                query_id=query_id, question=cleaned, role=role, style=style, status="declined_out_of_domain",
                message=OUT_OF_DOMAIN_MESSAGE, scope=scope, guard=guard, retrieval_query=retrieval_query,
                timings={"retrieval_s": round(retrieval_s, 3)}, domain_category="low_relevance",
                domain_action="relevance_declined", relevance_gate=gate), log)

        conflicts: list = []
        base = {"query_id": query_id, "question": cleaned, "retrieval_query": retrieval_query,
                "role": role, "style": style, "time": datetime.now().isoformat(timespec="seconds"),
                "access_notice": RESTRICTED_NOTICE if scope.restricted else None,
                "scope_rules": list(scope.matched_rules),
                "injection_rules": list(guard.rules), "request_signals": list(guard.signals),
                "injection_action": "shadow_would_decline" if guard.declined else None,
                "domain_category": domain.category, "domain_notice": domain_notice,
                "domain_action": ("partial" if domain_notice else
                                  "shadow_would_decline" if (domain.out_of_domain or domain.permitted_question) else None),
                "relevance_gate": gate,
                "statement_action": "shadow_would_decline" if statement else None,
                "premise_conflicts": conflicts}

        if not selected:  # never call the LLM with an empty context
            response = {**base, "answer": NO_ANSWER, "answer_display": NO_ANSWER, "sources": [],
                        "cited_sources": [], "status": "no_context", "evaluation": None,
                        "citations_valid": True, "invalid_citation_ids": [], "cited_ids": [],
                        "usage": {}, "timings": {"retrieval_s": round(retrieval_s, 3)}}
            return self._finish_declined(response, log)

        # 3b. Access verification BEFORE generation: if a retrieved chunk is outside the role's levels the
        #     model is never called and nothing is streamed.
        if not guardrails.verify_access(selected, levels, include_archived):
            return self._finish_declined({**base, "answer": ACCESS_BLOCKED_MESSAGE,
                                          "answer_display": ACCESS_BLOCKED_MESSAGE, "sources": [],
                                          "cited_sources": [], "status": "access_blocked", "evaluation": None,
                                          "citations_valid": True, "invalid_citation_ids": [], "cited_ids": [],
                                          "usage": {}, "timings": {"retrieval_s": round(retrieval_s, 3)}}, log)

        # 3c. A question built on a figure the documents contradict ("paternity leave is 7 weeks" when they say
        #     7 days) is not answered: the model would repeat or build on the wrong premise.
        conflicts = find_premise_conflicts(effective_question, sources) if self.cfg.premise_guard_mode != "off" else []
        if conflicts and self.cfg.premise_guard_mode == "enforce":
            return self._finish_declined(self._declined(
                query_id=query_id, question=cleaned, role=role, style=style, status="declined_premise",
                message=premise_message(conflicts), scope=scope, guard=guard, retrieval_query=retrieval_query,
                timings={"retrieval_s": round(retrieval_s, 3)}, premise_conflicts=conflicts), log)

        # 3d. Roster and headcount questions: when the department's complete roster chunk was retrieved and the
        #     question is only a roster request, the answer is written from that chunk (complete by construction).
        #     A roster-style question answered from individual rows only carries a note that it may be incomplete.
        roster_plan = (plan_roster_answer(effective_question, sources) if self.cfg.roster_lookup_mode != "off"
                       else RosterPlan())
        roster_text = roster_plan.text if self.cfg.roster_lookup_mode == "enforce" else None

        # 4. generate (every pre-generation check has passed; streaming starts only here)
        system_prompt = build_system_prompt(style)
        user_prompt = build_user_prompt(effective_question, context)
        # A source's own name and section title count as part of it ("Leave Policy v2" is grounded by leave_policy_v2.docx).
        source_text = "\n".join(f'{s["file_name"]} {s.get("section_title") or ""} {s["text"]}' for s in sources)
        numeric_mode = self.cfg.numeric_gate_mode
        gen_started = time.perf_counter()
        streaming = on_delta is not None and self.llm_stream is not None and roster_text is None
        stream_guard, callback = None, on_delta
        if streaming and numeric_mode != "off":
            # Text with a number is shown only after the number has been checked against the sources.
            stream_guard = NumericStreamGuard(
                on_delta, lambda text: numeric_issues(text, source_text, effective_question),
                enforce=numeric_mode == "enforce")
            callback = stream_guard.push
        try:
            if roster_text is not None:
                raw = (roster_text, {"input_tokens": 0, "output_tokens": 0})          # no model call
            else:
                raw = (self.llm_stream(system_prompt, user_prompt, callback) if streaming
                       else self.llm(system_prompt, user_prompt))
            if stream_guard:
                stream_guard.finish()
        except StreamAborted as stop:
            return self._withheld(base, stop.issues, numeric_mode, retrieval_s, time.perf_counter() - gen_started,
                                  estimate_tokens(system_prompt + user_prompt),
                                  estimate_tokens(stream_guard.raw if stream_guard else ""), len(sources), log,
                                  diagnostic_sources=sources)
        generation_s = time.perf_counter() - gen_started
        answer, usage = raw if isinstance(raw, tuple) else (raw, {})
        answer = answer.strip()

        # Numeric gate on the finished answer (also covers a model that cannot stream): invented, calculated or
        # unit-swapped figures are withheld, never shown.
        numeric = {"mode": numeric_mode,
                   "issues": numeric_issues(answer, source_text, effective_question) if numeric_mode != "off" else []}
        if numeric["issues"] and numeric_mode == "enforce":
            return self._withheld(base, numeric["issues"], numeric_mode, retrieval_s, generation_s,
                                  estimate_tokens(system_prompt + user_prompt), estimate_tokens(answer),
                                  len(sources), log, diagnostic_sources=sources)
        if on_delta is not None and not streaming:
            on_delta(answer)             # a model that cannot stream delivers the whole answer at once
        usage = {
            "input_tokens": usage.get("input_tokens") or estimate_tokens(system_prompt + user_prompt),
            "output_tokens": usage.get("output_tokens") or estimate_tokens(answer),
            "estimated": not (usage.get("input_tokens") and usage.get("output_tokens")),
            "context_chunks": len(sources),
        }
        if roster_text is not None:
            usage = {"input_tokens": 0, "output_tokens": 0, "estimated": False, "context_chunks": len(sources),
                     "generation": "roster_lookup"}

        # 5. guardrails (access was already verified before generation)
        citations_valid, invalid_ids, used_ids = guardrails.check_citations(answer, len(sources))
        status = "access_verified"

        # Secondary anomaly telemetry only: recorded, never used to change or withhold the answer.
        echo_anomaly = question_echo_anomaly(effective_question, answer, [s["text"] for s in sources])

        # 6. class, source cards and evaluation. Cards come only from a normal answer's own citations;
        #    there is no "top 3 sources" fallback, so a refusal can never look supported.
        response_class = assessment.classify(status, answer, used_ids)
        cards, unsupported_ids = assessment.select_source_cards(answer, sources, used_ids, response_class)
        report = None
        if response_class == assessment.NORMAL:
            report = evaluation.evaluate(
                question_embedding=self.embeddings.embed_query(retrieval_query),
                answer=answer, context=context, source_texts=[s["text"] for s in sources],
                sources=sources, status=status, citations_valid=citations_valid,
                embeddings=self.embeddings, cfg=self.cfg, question=effective_question)
        result = assessment.assess(response_class, llm_called=True, answer=answer, cards=cards,
                                   context_text=context, question=effective_question,
                                   fixed_messages=self._fixed_messages, unsupported_ids=unsupported_ids)
        response = {**base, "answer": answer, "answer_display": strip_citations(answer),
                    "sources": sources, "cited_sources": cards, "status": status,
                    "citations_valid": citations_valid, "invalid_citation_ids": invalid_ids,
                    "cited_ids": used_ids, "evaluation": report, "usage": usage, "echo_anomaly": echo_anomaly,
                    "response_class": response_class, "class_checks": result["checks"],
                    "class_passed": result["passed"], "unsupported_citation_ids": unsupported_ids,
                    "numeric_gate": numeric, "premise_conflicts": conflicts,
                    "roster_lookup": {"mode": self.cfg.roster_lookup_mode, "intent": roster_plan.intent,
                                      "departments": list(roster_plan.departments),
                                      "answered_from_roster": roster_text is not None},
                    "roster_notice": (ROSTER_NOTICE if roster_plan.notice and self.cfg.roster_lookup_mode == "enforce"
                                      else None),
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
