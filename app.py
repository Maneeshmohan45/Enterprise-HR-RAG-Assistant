"""HR RAG Assistant: Streamlit UI on top of the existing RAGPipeline.

    streamlit run app.py

Roles are SIMULATED (a dropdown). A real deployment must take the role from a login
system (SSO / JWT), never from something the user can change.
"""
from __future__ import annotations

import html
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from hr_rag import evaluation
from hr_rag.streaming import visible_text
from hr_rag.config import ROLE_ACCESS, get_api_key, settings
from hr_rag.manifest import load_manifest
from hr_rag.ui_data import (CHECK_LABELS, CLASS_LABELS, friendly_error, query_table_rows, read_jsonl,
                            redact, summarize_feedback, summarize_queries)

st.set_page_config(page_title="HR RAG Assistant", page_icon="👥", layout="wide")

# The backend still defines employee/manager/hr/admin (hr_rag.config.ROLE_ACCESS);
# this UI exposes only the two roles that are needed.
ROLES = [r for r in ("employee", "hr") if r in ROLE_ACCESS]
ROLE_LABELS = {"employee": "Employee", "hr": "HR"}
PAGES = {"assistant": "Assistant", "evaluation": "Model Evaluation", "about": "About"}
SUBTITLES = {
    "assistant": "Ask questions about HR policies, procedures and company information.",
    "evaluation": "Measured results from this app's own query log.",
    "about": "System architecture, data sources and technology details.",
}
FILE_BADGES = {".pdf": ("PDF", "#d64545"), ".docx": ("DOC", "#2f6fdd"), ".xlsx": ("XLS", "#1f9d6b"),
               ".csv": ("CSV", "#1f9d6b"), ".json": ("JSON", "#8a63d2"),
               ".html": ("HTML", "#e08a1e"), ".txt": ("TXT", "#6b7a90")}

CSS = """
<style>
/* ---- page frame: tighten the default Streamlit top padding ---- */
[data-testid="stHeader"] {background:transparent;}
[data-testid="stHeader"], [data-testid="stHeader"] * {pointer-events:none;}   /* never cover the page controls */
[data-testid="stHeader"] button, [data-testid="stHeader"] button * {pointer-events:auto;}   /* native expand-sidebar button */
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
[data-testid="stApp"]:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"]
    {padding-left:4rem !important;}   /* only while the sidebar is collapsed: clears the native expand button */
[data-testid="stMainBlockContainer"], .block-container {padding:1.4rem 2rem 1rem 2rem !important;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {gap:.75rem;}
hr {margin:.6rem 0 !important;}
.page-title {font-size:1.6rem; font-weight:700; color:#13264d; line-height:1.15; margin:0;}
.page-sub {font-size:.9rem; color:#6b7a90; margin:.15rem 0 0 0;}
.ctl-label {font-size:.8rem; color:#6b7a90; white-space:nowrap;}

/* ---- header controls: neutral, polished (no red accent) ---- */
[data-baseweb="select"] > div {border-color:#cfd8e8 !important; background:#fff; border-radius:10px;
    min-height:2.4rem;}
[data-baseweb="select"] > div:hover {border-color:#9db4dd !important;}
[data-baseweb="select"] > div:focus-within {border-color:#1d4fa8 !important; box-shadow:0 0 0 1px #1d4fa8;}
[data-testid="stSelectbox"] {margin:0;}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {background:#0f1f3d;}
[data-testid="stSidebar"] * {color:#e8eefc;}
[data-testid="stSidebarHeader"] {height:2rem; min-height:0; padding:.5rem .75rem 0 .75rem;}
[data-testid="stSidebarUserContent"] {padding:.25rem 1rem 1rem 1rem !important;}
.st-key-sidebar_shell {min-height:calc(100vh - 4.5rem); gap:.35rem;}
.st-key-sidebar_shell > div:has(.st-key-sidebar_bottom), .st-key-sidebar_shell > .st-key-sidebar_bottom
    {margin-top:auto; position:sticky; bottom:0; background:#0f1f3d; z-index:2;}
.st-key-sidebar_bottom {gap:.4rem; border-top:1px solid #243a6b; padding-top:.7rem;}
[data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] [data-testid="stPopoverButton"]
    {width:100%; justify-content:flex-start; text-align:left; background:transparent;
    border:1px solid transparent; border-radius:10px; padding:.5rem .8rem; min-height:2.4rem;}
[data-testid="stSidebar"] .stButton > button:hover, [data-testid="stSidebar"] [data-testid="stPopoverButton"]:hover
    {background:#1b3160; border-color:#2b4a8a;}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {background:#1d4fa8; border-color:#1d4fa8;}
.st-key-nav_new button {border:1px solid #3a5088 !important;}
.brand {display:flex; gap:.7rem; align-items:center; margin:0 0 .6rem 0;}
.brand .logo {background:#1f6bff; border-radius:12px; width:40px; height:40px; display:flex;
    align-items:center; justify-content:center; font-size:1.3rem;}
.brand .name {font-weight:700; font-size:1.05rem; line-height:1.1;}
.brand .tag {font-size:.72rem; color:#aebbd8;}
.dev-by {font-size:.78rem; color:#aebbd8; padding:.1rem .8rem 0 .8rem;}

/* ---- chat ---- */
.user-bubble {background:#dfeafd; border-radius:14px; padding:.7rem 1rem; margin:.4rem 0 .2rem auto;
    max-width:78%; width:fit-content; color:#13264d;}
.msg-time {font-size:.72rem; color:#7a869c; text-align:right;}
.src-card {border:1px solid #d9e0ec; border-radius:12px; padding:.6rem .8rem; display:flex; gap:.7rem;
    align-items:center; background:#fff; margin-bottom:.5rem;}
.src-badge {color:#fff; font-weight:700; font-size:.68rem; border-radius:8px; padding:.45rem .4rem;
    min-width:2.6rem; text-align:center;}
.src-title {font-weight:600; color:#13264d; line-height:1.2;}
.src-meta {font-size:.8rem; color:#6b7a90;}
.src-old {background:#fff1d6; color:#8a5a00; border-radius:6px; padding:0 .4rem; font-size:.7rem; margin-left:.4rem;}
.meta-line {font-size:.78rem; color:#6b7a90; margin-top:.3rem;}
.conv-title {font-size:1.1rem; font-weight:700; color:#13264d;}
</style>
"""


# ---------------------------------------------------------------------------
# Start-up: index check, API-key check, cached pipeline
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading search models (first start only)...")
def load_pipeline():
    from langchain_core.documents import Document

    from hr_rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings)
    # Warm up so the first real question does not pay for loading the models.
    pipeline.embeddings.embed_query("warm up")
    pipeline.retriever.reranker.score("warm up", [Document(page_content="warm up")])
    return pipeline


def get_pipeline():
    """Return the pipeline, or show a clear message and stop the app."""
    if not settings.chroma_dir.exists():
        st.error("No document index found. Build it first by running `python ingest.py` "
                 "in your project folder, then refresh this page.")
        st.stop()
    try:
        return load_pipeline()
    except Exception as error:  # empty index, missing model files, ...
        message, detail = friendly_error(error)
        st.error(message)
        with st.expander("Details"):
            st.code(detail)
        st.stop()


def llm_config_problem() -> str | None:
    """A message if the LLM cannot be called (missing key etc.). The key is never read into the UI."""
    try:
        get_api_key(settings.llm_provider)
    except ValueError as error:
        return friendly_error(error)[0]
    return None


def active_model_name() -> str:
    return settings.nvidia_model if settings.llm_provider == "nvidia" else settings.anthropic_model


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    default_style = settings.answer_style if settings.answer_style in ("concise", "detailed") else "concise"
    st.session_state.setdefault("page", "assistant")
    st.session_state.setdefault("conversations", [])   # newest first, kept in memory only
    st.session_state.setdefault("active_id", None)
    st.session_state.setdefault("role", "employee")
    if st.session_state.role not in ROLES:
        st.session_state.role = "employee"
    st.session_state.setdefault("pending_role", None)   # a role change waiting for confirmation
    if st.session_state.get("role_select") not in ROLES:
        st.session_state.role_select = st.session_state.role   # the dropdown mirrors the confirmed role
    st.session_state.setdefault("include_archived", False)
    st.session_state.setdefault("style", default_style)
    st.session_state.setdefault("conv_search", "")


def new_conversation() -> dict:
    conv = {"id": uuid.uuid4().hex[:8], "title": "New chat", "created": datetime.now(), "messages": []}
    st.session_state.conversations.insert(0, conv)
    st.session_state.active_id = conv["id"]
    return conv


ROLE_SWITCH_WARNING = ("⚠️ Switching roles will clear your current conversation. "
                       "Your previous conversation will not be available after switching roles. Continue?")


def reset_conversations() -> None:
    """Forget every chat of this session: messages, answers, sources, feedback and the search box."""
    st.session_state.conversations = []
    st.session_state.active_id = None
    st.session_state.conv_search = ""
    # The per-message widgets (👍 / 👎, conversation buttons) are discarded by Streamlit as soon as
    # they stop being rendered, so clearing the conversations also clears their state.


def request_role_change() -> None:
    """The dropdown moved: remember the wish, but change nothing until the user confirms."""
    chosen = st.session_state.role_select
    st.session_state.pending_role = chosen if chosen != st.session_state.role else None


def confirm_role_change() -> None:
    """Switch role and start completely fresh: there is no per-role chat history."""
    new_role = st.session_state.pending_role
    if new_role in ROLES:
        st.session_state.role = new_role
        reset_conversations()
    st.session_state.role_select = st.session_state.role
    st.session_state.pending_role = None


def cancel_role_change() -> None:
    """Keep the current role and conversation exactly as they are."""
    st.session_state.role_select = st.session_state.role
    st.session_state.pending_role = None


def active_conversation() -> dict:
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.active_id:
            return conv
    return new_conversation()


def clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        with st.container(key="sidebar_shell"):
            st.markdown('<div class="brand"><div class="logo">👥</div><div><div class="name">HR RAG Assistant</div>'
                        '<div class="tag">Your company knowledge assistant</div></div></div>',
                        unsafe_allow_html=True)
            if st.button("＋  New Chat", key="nav_new"):
                current = [c for c in st.session_state.conversations if c["id"] == st.session_state.active_id]
                if not (current and not current[0]["messages"]):
                    new_conversation()
                st.session_state.page = "assistant"
                st.rerun()
            for key, label in PAGES.items():
                if st.button(label, key=f"nav_{key}",
                             type="primary" if st.session_state.page == key else "secondary"):
                    st.session_state.page = key
                    st.rerun()
            with st.container(key="sidebar_bottom"):   # pinned to the bottom of the sidebar by CSS
                with st.popover("⚙  Settings"):
                    st.radio("Answer style", ["concise", "detailed"], key="style",
                             help="Concise: short, simple answers. Detailed: fuller explanations (uses more tokens).")
                    st.caption(f"Model: {active_model_name()}")
                st.markdown('<div class="dev-by">Developed by <b>Maneesh Mohan</b></div>',
                            unsafe_allow_html=True)


def render_header() -> None:
    page = st.session_state.page
    left, role_col, arch_col = st.columns([3, 1.5, 1.7], vertical_alignment="center")
    with left:
        st.markdown(f'<div class="page-title">{PAGES[page]}</div><div class="page-sub">{SUBTITLES[page]}</div>',
                    unsafe_allow_html=True)
    with role_col:
        label, control = st.columns([0.35, 1], vertical_alignment="center")
        label.markdown('<span class="ctl-label">Role</span>', unsafe_allow_html=True)
        control.selectbox("Role", ROLES, key="role_select", on_change=request_role_change,
                          format_func=ROLE_LABELS.get, label_visibility="collapsed",
                          help="Simulated role: there is no login. It decides which documents can be searched.")
    with arch_col:
        st.toggle("Include archived policies", key="include_archived",
                  help="Also search old, superseded policy versions. Answers will label them as archived.")
    if st.session_state.pending_role and st.session_state.pending_role != st.session_state.role:
        st.warning(ROLE_SWITCH_WARNING)
        confirm_col, cancel_col, _ = st.columns([1, 1, 6])
        confirm_col.button("Continue", key="role_confirm", type="primary", on_click=confirm_role_change)
        cancel_col.button("Cancel", key="role_cancel", on_click=cancel_role_change)
    st.divider()


def source_card_html(source: dict) -> str:
    ext = Path(source.get("file_name") or "").suffix.lower()
    label, color = FILE_BADGES.get(ext, ("FILE", "#6b7a90"))
    where = " · ".join(x for x in (source.get("section_title"), source.get("location")) if x)
    old = '<span class="src-old">ARCHIVED</span>' if not source.get("is_current", True) else ""
    return (f'<div class="src-card"><div class="src-badge" style="background:{color}">{label}</div>'
            f'<div><div class="src-title">{html.escape(str(source.get("title") or source.get("file_name") or ""))}{old}</div>'
            f'<div class="src-meta">{html.escape(where)}</div>'
            f'<div class="src-meta">{html.escape(str(source.get("file_name") or ""))}</div></div></div>')


def render_sources(sources: list[dict], key: str) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=True):
        columns = st.columns(2)
        for i, source in enumerate(sources):
            columns[i % 2].markdown(source_card_html(source), unsafe_allow_html=True)
        with st.expander("Show the text each source contributed"):
            for i, source in enumerate(sources, start=1):
                st.markdown(f"**{source.get('title') or source.get('file_name')}** "
                            f"· {source.get('section_title') or ''}")
                st.text(str(source.get("text", ""))[:1200])


def render_quality(response: dict) -> None:
    report, usage, timings = response.get("evaluation"), response.get("usage") or {}, response.get("timings") or {}
    parts = []
    if report:
        parts.append(f"Quality check {report['evaluation_score']:.0f}%")
    if usage.get("output_tokens"):
        mark = "~" if usage.get("estimated") else ""
        parts.append(f"{mark}{usage['output_tokens']:,} tokens out")
    if timings.get("total_s") is not None:
        parts.append(f"{timings['total_s']:.1f}s")
    if parts:
        st.markdown(f'<div class="meta-line">{" · ".join(parts)}</div>', unsafe_allow_html=True)
    cls = response.get("response_class")
    if cls and cls != "NORMAL_GROUNDED" and response.get("class_passed") is not None:
        mark = "✓" if response["class_passed"] else "✗ check failed"
        st.caption(f"Handled as: {CLASS_LABELS.get(cls, cls)} {mark}")
    if response.get("unsupported_citation_ids"):
        st.caption("A cited source was hidden because it does not support the answer: "
                   + ", ".join(response["unsupported_citation_ids"]))
    if report:
        failed = [CHECK_LABELS.get(k, k) for k, v in report.items() if isinstance(v, bool) and not v]
        if failed:
            st.caption("Checks not passed: " + ", ".join(failed))
        if report.get("ungrounded_numbers"):
            st.warning("These numbers in the answer were not found in the sources, so please verify them: "
                       + ", ".join(report["ungrounded_numbers"]))


def record_feedback_for(message: dict, positive: bool) -> None:
    try:
        evaluation.record_feedback(message["response"], 5 if positive else 1, positive, "", settings)
        message["feedback"] = "up" if positive else "down"
    except Exception as error:  # never break the chat over a logging problem
        st.toast(f"Could not save feedback: {redact(error)}")


def render_message(message: dict) -> None:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{html.escape(message["content"])}</div>'
                    f'<div class="msg-time">{message["time"]}</div>', unsafe_allow_html=True)
        return
    with st.chat_message("assistant", avatar="🤖"):
        if message.get("error"):
            st.error(message["content"])
            if message.get("detail"):
                with st.expander("Details"):
                    st.code(message["detail"])
            return
        response = message["response"]
        if response["status"] in ("access_blocked", "access_restricted", "declined_injection"):
            st.warning(message["content"])
        elif response["status"] in ("no_context", "declined_out_of_domain", "declined_ungrounded"):
            st.info(message["content"])
        else:
            st.markdown(message["content"])
        render_quality(response)
        render_sources(response.get("cited_sources") or [], key=message["id"])
        if response.get("access_notice"):
            st.warning(response["access_notice"])
        if response.get("domain_notice"):
            st.warning(response["domain_notice"])
        if response["status"] == "access_verified":
            if message.get("feedback"):
                st.caption("Thanks for the feedback.")
            else:
                up, down, _ = st.columns([1, 1, 12])
                if up.button("👍", key=f"up_{message['id']}", help="Helpful"):
                    record_feedback_for(message, True)
                    st.rerun()
                if down.button("👎", key=f"down_{message['id']}", help="Not helpful"):
                    record_feedback_for(message, False)
                    st.rerun()
        st.markdown(f'<div class="msg-time">{message["time"]}</div>', unsafe_allow_html=True)


def ask_pipeline(pipeline, question: str) -> dict:
    """Run one question and return the chat message for the answer (or for the error)."""
    now = clock(datetime.now())
    # The live answer sits in a temporary holder that is emptied once the final message is rendered, so
    # nothing is shown twice. Text arrives only after every pre-generation check has passed.
    holder = st.empty()
    try:
        with holder.container():
            with st.chat_message("assistant", avatar="🤖"):
                box = st.empty()
                box.markdown("🔎 _Searching the HR documents..._")
                raw = []

                def on_delta(piece: str) -> None:
                    raw.append(piece)
                    box.markdown(visible_text("".join(raw)) + " ▌")

                response = pipeline.ask(
                    question, {"user_id": "streamlit_user", "role": st.session_state.role},
                    include_archived=st.session_state.include_archived, style=st.session_state.style,
                    on_delta=on_delta if settings.stream_responses else None)
    except Exception as error:  # ValueError, PermissionError, LLMError, anything else
        holder.empty()
        text, detail = friendly_error(error)
        return {"id": uuid.uuid4().hex, "role": "assistant", "content": text, "detail": detail,
                "error": True, "time": now}
    holder.empty()
    return {"id": uuid.uuid4().hex, "role": "assistant", "content": response["answer_display"],
            "response": response, "feedback": None, "time": now}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_conversation_list() -> None:
    st.markdown('<div class="conv-title">Conversations</div>', unsafe_allow_html=True)
    st.text_input("Search conversations", key="conv_search", placeholder="Search conversations...",
                  label_visibility="collapsed")
    needle = st.session_state.conv_search.strip().lower()
    shown = [c for c in st.session_state.conversations
             if c["messages"] and (not needle or needle in c["title"].lower())]
    if not shown:
        st.caption("No conversations yet." if not needle else "No matches.")
    else:
        st.caption("This session")
    for conv in shown:
        label = f"{conv['title'][:34]}{'…' if len(conv['title']) > 34 else ''}   ·   {clock(conv['created'])}"
        if st.button(label, key=f"conv_{conv['id']}",
                     type="primary" if conv["id"] == st.session_state.active_id else "secondary"):
            st.session_state.active_id = conv["id"]
            st.rerun()
    if shown and st.button("🗑  Clear all conversations", key="clear_all"):
        st.session_state.conversations = []
        st.session_state.active_id = None
        st.rerun()
    st.caption("Conversations are kept only while this browser tab is open. "
               "Each question is answered on its own, without earlier messages as context.")


def assistant_page(pipeline) -> None:
    problem = llm_config_problem()
    if problem:
        st.error(problem)
    list_col, chat_col = st.columns([1, 3], gap="large")
    with list_col:
        render_conversation_list()
    prompt = st.chat_input("Ask a question about HR policies, procedures or company information...",
                           disabled=bool(problem) or bool(st.session_state.pending_role))
    with chat_col:
        conv = active_conversation()
        if not conv["messages"] and not prompt:
            st.info(f"Ask about leave, attendance, onboarding, work from home and more. "
                    f"You are searching as **{st.session_state.role}**.")
        for message in conv["messages"]:
            render_message(message)
        if prompt:
            user_message = {"id": uuid.uuid4().hex, "role": "user", "content": prompt.strip(),
                            "time": clock(datetime.now())}
            render_message(user_message)
            answer_message = ask_pipeline(pipeline, prompt)
            if not conv["messages"]:
                conv["title"] = " ".join(prompt.split())[:80]
            conv["messages"] += [user_message, answer_message]
            render_message(answer_message)


def evaluation_page() -> None:
    rows = read_jsonl(settings.logs_dir / "query_log.jsonl")
    feedback = read_jsonl(settings.logs_dir / "feedback.jsonl")
    summary = summarize_queries(rows)
    if not rows:
        st.info("No queries logged yet. Ask a question on the Assistant page and the results will appear here.")
        return

    def fmt(value, template="{:.1f}", missing="n/a"):
        return template.format(value) if value is not None else missing

    st.caption("Numbers come from `logs/query_log.jsonl` only. Older log rows may lack tokens, timings or "
               "per-check results; those rows are left out of the averages rather than guessed.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries logged", summary["n_queries"])
    c2.metric("Avg quality score", fmt(summary["avg_score"], "{:.1f}%"),
              help="Share of automatic checks passed (grounding, citations, access). It does not prove "
                   "that an answer is factually correct or complete.")
    c3.metric("Avg response time", fmt(summary["avg_total_s"], "{:.1f}s"),
              help=f"Based on {summary['n_timed']} queries that recorded timings.")
    c4.metric("Avg tokens in / out",
              f"{fmt(summary['avg_input_tokens'], '{:,.0f}')} / {fmt(summary['avg_output_tokens'], '{:,.0f}')}",
              help=f"Based on {summary['n_with_usage']} queries that recorded token usage.")

    tab_quality, tab_latency, tab_checks, tab_log = st.tabs(
        ["Answer quality", "Latency and tokens", "Checks", "Query log"])

    with tab_quality:
        import pandas as pd
        scored = [{"Query": i + 1, "Quality score %": r["evaluation_score"]}
                  for i, r in enumerate(rows) if r.get("evaluation_score") is not None]
        if scored:
            st.markdown("**Quality score per query (oldest to newest)**")
            st.line_chart(pd.DataFrame(scored).set_index("Query"))
        st.markdown("**Responses by class**")
        class_rows = []
        for cls, n in sorted(summary["class_counts"].items(), key=lambda kv: -kv[1]):
            integrity = summary["class_integrity"].get(cls)
            class_rows.append({"Class": CLASS_LABELS.get(cls, cls), "Responses": n,
                               "Handled correctly": (f"{integrity['passed']}/{integrity['total']}"
                                                     if integrity else ("quality score" if cls == "NORMAL_GROUNDED" else "n/a"))})
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)
        st.caption("Only normal answers get a quality score. A refusal or a not-found answer is checked for "
                   "its own behaviour (nothing invented, no source cards, fixed message) instead. These checks "
                   "show the system behaved as designed; they cannot prove a refusal was warranted.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Status of answers**")
            st.bar_chart(pd.DataFrame({"Queries": summary["status_counts"]}))
        with right:
            st.markdown("**User feedback (thumbs and CLI ratings)**")
            fb = summarize_feedback(feedback)
            if fb["n"]:
                st.metric("Feedback entries", fb["n"])
                st.write(f"Helpful: **{fb['helpful']}** · Not helpful: **{fb['not_helpful']}** · "
                         f"Avg rating: **{fmt(fb['avg_rating'], '{:.1f}')}**")
            else:
                st.caption("No feedback recorded yet.")
        st.info("Retrieval metrics such as Recall@k and MRR are not shown: they need a labeled set of "
                "questions with known correct sources, and none exists yet.")

    with tab_latency:
        import pandas as pd
        timed = [{"Query": i + 1, "Retrieval (s)": r["timings"].get("retrieval_s"),
                  "Generation (s)": r["timings"].get("generation_s")}
                 for i, r in enumerate(rows) if (r.get("timings") or {}).get("total_s") is not None]
        if timed:
            st.markdown("**Seconds per query, split into retrieval and generation**")
            st.bar_chart(pd.DataFrame(timed).set_index("Query"))
            st.caption(f"Average retrieval {fmt(summary['avg_retrieval_s'])}s, generation "
                       f"{fmt(summary['avg_generation_s'])}s over {summary['n_timed']} timed queries. "
                       "Runs from the command line also pay for loading the models; this app loads them once.")
        else:
            st.caption("No timings recorded yet.")
        if summary["by_style"]:
            st.markdown("**Average tokens by answer style**")
            st.dataframe(pd.DataFrame(summary["by_style"]).T.rename(columns={
                "n": "Queries", "avg_input_tokens": "Avg tokens in",
                "avg_output_tokens": "Avg tokens out", "avg_total_s": "Avg seconds"}).round(1))
            st.caption("Averages over different questions are not a like-for-like comparison. "
                       "Use the same question in both styles to compare them.")

    with tab_checks:
        import pandas as pd
        if summary["n_with_checks"]:
            st.markdown(f"**Pass rate per check** (over {summary['n_with_checks']} queries that recorded checks)")
            rates = {CHECK_LABELS[k]: round(v * 100, 1) for k, v in summary["check_pass_rates"].items()}
            st.bar_chart(pd.DataFrame({"Pass rate %": rates}))
        else:
            st.info("Per-check results are recorded for new queries only. Ask a few questions and they "
                    "will appear here.")
        st.caption("Grounding checks compare the answer with the retrieved text. They catch unsupported "
                   "statements and invented numbers, but a grounded answer can still be incomplete.")

    with tab_log:
        show_questions = st.session_state.role in ("hr", "admin")
        if not show_questions:
            st.caption("Question text is hidden for this role because questions can mention confidential records.")
        st.dataframe(query_table_rows(rows, show_questions), hide_index=True)


def about_page() -> None:
    tabs = st.tabs(["Overview", "Architecture", "Data Sources", "Usage Guide", "Limitations"])
    with tabs[0]:
        st.markdown("### HR RAG Assistant")
        st.write("An AI assistant that answers employee HR questions from the company's own documents, "
                 "with source citations, role-based access to confidential data, and archived policy "
                 "versions hidden by default.")
        a, b, c, d = st.columns(4)
        a.markdown("**Multi-format ingestion**  \nPDF, DOCX, TXT, HTML, CSV, JSON and XLSX, with validation.")
        b.markdown("**Hybrid search**  \nVector search plus keyword (BM25), fused, then reranked.")
        c.markdown("**Grounded answers**  \nCitations and automatic checks against the sources.")
        d.markdown("**Role-based access**  \nRestricted text is filtered out before the model sees anything.")
        st.markdown("#### Technology stack")
        t1, t2, t3, t4 = st.columns(4)
        t1.markdown("**Vector database**  \nChromaDB")
        t2.markdown(f"**Embeddings**  \n{settings.embedding_model.split('/')[-1]}")
        t3.markdown(f"**Language model**  \n{active_model_name()} ({settings.llm_provider})")
        t4.markdown(f"**Reranker**  \n{settings.reranker_model.split('/')[-1]}")
    with tabs[1]:
        st.markdown(
            "**Ingestion (`python ingest.py`)**  \n"
            "Load each file with its own reader, split into sections, clean, validate, remove duplicates "
            "(current files win over archived ones), enrich with metadata from `manifest.yaml`, chunk "
            f"({settings.chunk_size} characters, {settings.chunk_overlap} overlap) and store in ChromaDB.\n\n"
            "**Answering a question**  \n"
            "1. Check the question and the role.  \n"
            "2. Expand abbreviations and split multi-part questions.  \n"
            "3. Search only the documents the role may see: the permission filter runs inside the database "
            "query, and a matching filter is applied to keyword search.  \n"
            "4. Combine vector and keyword results (Reciprocal Rank Fusion), then rerank with a cross-encoder.  \n"
            f"5. Build the context: up to {settings.context_chunks} chunks for simple questions, "
            f"{settings.context_chunks_complex} for multi-part or list/compare questions.  \n"
            "6. Generate a cited answer, then run the checks: citation validity, grounding, relevance, "
            "figures not found in the sources, and an access re-check.  \n"
            "7. Log the query (role, style, tokens, timings, score).")
    with tabs[2]:
        manifest = load_manifest(settings.manifest_path)
        defaults, files = manifest.get("defaults", {}), manifest.get("files", {})
        allowed = ROLE_ACCESS.get(st.session_state.role, [])
        table = []
        for name, entry in files.items():
            entry = {**defaults, **(entry or {})}
            if entry.get("exclude") or entry.get("access_level", "internal") not in allowed:
                continue
            table.append({"File": name, "Title": entry.get("title", name),
                          "Access": entry.get("access_level", "internal"),
                          "Version": entry.get("version_number", 1),
                          "Current": "yes" if entry.get("is_current", True) else "archived"})
        st.write(f"Documents the **{st.session_state.role}** role can search "
                 f"({len(table)} of {len(files)} listed). Change the role at the top to see how this changes.")
        st.dataframe(table, hide_index=True)
        st.markdown("**Roles (simulated):** Employee sees public and internal documents; HR also sees "
                    "confidential ones. The backend also defines manager and admin roles, which this UI does "
                    "not expose.")
    with tabs[3]:
        st.markdown(
            "- Pick a **role** at the top to simulate who is asking.  \n"
            "- Turn on **Include archived policies** to compare with old policy versions.  \n"
            "- Under **Settings** choose **concise** (short answers) or **detailed**.  \n"
            "- Cited sources appear under each answer. Open one to see the exact text used.  \n"
            "- Use 👍 / 👎 to rate an answer; it is saved to `logs/feedback.jsonl` "
            "(👍 is stored as rating 5, 👎 as rating 1).  \n"
            "- If an answer contains a figure that is not in the sources, a warning is shown.")
    with tabs[4]:
        st.markdown(
            "- **Roles are simulated.** There is no login; a real deployment needs SSO or similar.  \n"
            "- **HR authentication.** The current version does not require a passcode to access the HR role. "
            "This is intentional for the testing and demonstration phase, so that users can evaluate the "
            "complete Employee and HR functionality. **Planned for V2:** HR role access will be protected by "
            "authentication/passcode so that confidential HR functionality cannot be accessed by unauthorized "
            "users. The underlying role-based retrieval controls still keep the Employee role away from "
            "confidential HR data.  \n"
            "- **Counting across rows is unreliable.** For example, counting employees in a department "
            "from a CSV can be wrong.  \n"
            "- **\"List all N items\" questions can be incomplete or padded.**  \n"
            "- **The model's explanations can be wrong even when the facts are right.**  \n"
            "- **The quality score measures grounding, not correctness.** A 100% score does not prove an "
            "answer is right or complete.  \n"
            "- **Each question is answered independently**; follow-up questions do not see earlier messages.  \n"
            "- **No labeled retrieval benchmark exists**, so Recall@k and MRR are not reported.  \n"
            "- **Latency is high for a chat UI** (see Model Evaluation for measured times).  \n"
            "- The synthetic dataset has 15 files; results may differ on real company documents.")


# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    pipeline = get_pipeline()
    render_sidebar()
    render_header()
    page = st.session_state.page
    if page == "assistant":
        assistant_page(pipeline)
    elif page == "evaluation":
        evaluation_page()
    else:
        about_page()


main()
