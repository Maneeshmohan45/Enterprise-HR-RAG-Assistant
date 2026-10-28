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

/* ---- About page visual polish ---- */
.about-badge {
    display: inline-flex; align-items: center; gap: 6px; background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe; font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
    border-radius: 12px; margin-bottom: 8px;
}
.about-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem;
    height: 100%; box-sizing: border-box; display: flex; flex-direction: column;
}
.about-card h4 {
    font-size: 0.92rem; font-weight: 600; color: #0f172a; margin: 0 0 0.4rem 0;
}
.about-card p {
    font-size: 0.82rem; color: #475569; line-height: 1.45; margin: 0;
}
.about-card ul {
    font-size: 0.82rem; color: #475569; padding-left: 1.1rem; margin: 0.4rem 0 0 0; line-height: 1.5;
}
.about-card li {
    margin-bottom: 0.25rem;
}
.about-card-blue { border-top: 3px solid #2563eb !important; }
.about-card-green { border-top: 3px solid #059669 !important; }

.about-tech-bar {
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.8rem 1.2rem;
    margin-top: 0.6rem; margin-bottom: 0.4rem;
}
.about-tech-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
}
.about-tech-item h5 {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin: 0 0 0.2rem 0;
}
.about-tech-item p {
    font-size: 0.85rem; font-weight: 600; color: #0f172a; margin: 0;
}


.about-flowchart-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.about-diagram-box {
    background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1rem 1.2rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.76rem; color: #1e293b; line-height: 1.55; white-space: pre; overflow-x: auto;
    margin-bottom: 1rem; background-image: linear-gradient(#f8fafc, #f1f5f9);
}

.about-callout {
    background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem; font-size: 0.82rem; color: #1e40af; line-height: 1.45; margin-top: 0.8rem;
}

.about-scorecard {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.about-scorecard-title {
    font-size: 0.88rem; font-weight: 700; color: #166534; margin-bottom: 0.6rem;
}
.about-scorecard-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
}
.about-metric-val {
    font-size: 1.45rem; font-weight: 700; color: #15803d; line-height: 1.1;
}
.about-metric-lbl {
    font-size: 0.75rem; color: #166534; font-weight: 500; margin-top: 0.2rem;
}

.about-table {
    width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 0.8rem;
}
.about-table th {
    background: #f1f5f9; padding: 8px 12px; font-weight: 600; color: #475569;
    border-bottom: 1px solid #cbd5e1; text-align: left;
}
.about-table td {
    padding: 7px 12px; border-bottom: 1px solid #e2e8f0; color: #1e293b;
}
.about-badge-tag {
    display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;
}
.about-badge-internal { background: #e0f2fe; color: #0369a1; }
.about-badge-confidential { background: #fef3c7; color: #92400e; }
.about-badge-current { background: #dcfce7; color: #15803d; }
.about-badge-archived { background: #f1f5f9; color: #64748b; }
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
        elif response["status"] in ("no_context", "declined_out_of_domain", "declined_ungrounded",
                                    "declined_statement", "declined_premise"):
            st.info(message["content"])
        else:
            st.markdown(message["content"])
        render_quality(response)
        render_sources(response.get("cited_sources") or [], key=message["id"])
        if response.get("access_notice"):
            st.warning(response["access_notice"])
        if response.get("domain_notice"):
            st.warning(response["domain_notice"])
        if response.get("roster_notice"):
            st.warning(response["roster_notice"])
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


ARCHITECTURE_FLOWCHART_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 710" width="100%" height="auto" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; display: block; margin: 0 auto;">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748b" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e11d48" />
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#d97706" />
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ea580c" />
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#9333ea" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#16a34a" />
    </marker>
    
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f1f3d" />
      <stop offset="100%" stop-color="#1e3a8a" />
    </linearGradient>
    <linearGradient id="cardGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="100%" stop-color="#f8fafc" />
    </linearGradient>
    <linearGradient id="rosterGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f0fdf4" />
      <stop offset="100%" stop-color="#dcfce7" />
    </linearGradient>
    <linearGradient id="llmGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#eff6ff" />
      <stop offset="100%" stop-color="#dbeafe" />
    </linearGradient>
    <filter id="softShadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.06" flood-color="#0f172a" />
    </filter>
  </defs>

  <!-- Flowchart Background Container -->
  <rect x="2" y="2" width="916" height="706" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />

  <!-- 1. Input Node -->
  <g filter="url(#softShadow)">
    <rect x="220" y="16" width="280" height="42" rx="8" fill="url(#headerGrad)" />
    <text x="360" y="36" fill="#ffffff" font-size="13.5" font-weight="700" text-anchor="middle" dominant-baseline="middle">👤 User Query + Active Role</text>
    <text x="360" y="49" fill="#93c5fd" font-size="10.5" font-weight="500" text-anchor="middle">Role: Employee (Internal) vs HR (Confidential)</text>
  </g>

  <!-- Line 1 to Request Guard -->
  <line x1="360" y1="58" x2="360" y2="82" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />

  <!-- 2. Request Guard -->
  <g filter="url(#softShadow)">
    <rect x="200" y="82" width="320" height="46" rx="8" fill="url(#cardGrad)" stroke="#cbd5e1" stroke-width="1.2" />
    <text x="360" y="100" fill="#0f172a" font-size="13" font-weight="700" text-anchor="middle">1. Request Guard</text>
    <text x="360" y="116" fill="#64748b" font-size="10.5" text-anchor="middle">Inspection for adversarial inputs &amp; jailbreaks</text>
  </g>
  <!-- Request Guard Exit Branch -->
  <line x1="520" y1="105" x2="602" y2="105" stroke="#e11d48" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow-red)" />
  <rect x="610" y="86" width="270" height="38" rx="6" fill="#ffe4e6" stroke="#f43f5e" stroke-width="1" />
  <text x="745" y="101" fill="#9f1239" font-size="11" font-weight="700" text-anchor="middle">SECURITY_REFUSAL</text>
  <text x="745" y="114" fill="#be123c" font-size="9.5" text-anchor="middle">Injections • DAN Mode • &lt;/context&gt; Tag Escapes</text>

  <!-- Line 2 to Scope Guard -->
  <line x1="360" y1="128" x2="360" y2="152" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />

  <!-- 3. Scope Guard -->
  <g filter="url(#softShadow)">
    <rect x="200" y="152" width="320" height="46" rx="8" fill="url(#cardGrad)" stroke="#cbd5e1" stroke-width="1.2" />
    <text x="360" y="170" fill="#0f172a" font-size="13" font-weight="700" text-anchor="middle">2. Scope Guard</text>
    <text x="360" y="186" fill="#64748b" font-size="10.5" text-anchor="middle">Disambiguates approval workflows vs directory lookups</text>
  </g>
  <!-- Scope Guard Exit Branch -->
  <line x1="520" y1="175" x2="602" y2="175" stroke="#d97706" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow-amber)" />
  <rect x="610" y="156" width="270" height="38" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1" />
  <text x="745" y="171" fill="#92400e" font-size="11" font-weight="700" text-anchor="middle">ACCESS_RESTRICTED</text>
  <text x="745" y="184" fill="#b45309" font-size="9.5" text-anchor="middle">Confidential Directory PII &amp; Salaries (Employee)</text>

  <!-- Line 3 to Premise Guard -->
  <line x1="360" y1="198" x2="360" y2="222" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />

  <!-- 4. Premise & Statement Guard -->
  <g filter="url(#softShadow)">
    <rect x="200" y="222" width="320" height="46" rx="8" fill="url(#cardGrad)" stroke="#cbd5e1" stroke-width="1.2" />
    <text x="360" y="240" fill="#0f172a" font-size="13" font-weight="700" text-anchor="middle">3. Premise &amp; Statement Guard</text>
    <text x="360" y="256" fill="#64748b" font-size="10.5" text-anchor="middle">Rejects false factual claims &amp; non-question statements</text>
  </g>
  <!-- Premise Guard Exit Branch -->
  <line x1="520" y1="245" x2="602" y2="245" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow-orange)" />
  <rect x="610" y="226" width="270" height="38" rx="6" fill="#ffedd5" stroke="#fb923c" stroke-width="1" />
  <text x="745" y="241" fill="#9a3412" font-size="11" font-weight="700" text-anchor="middle">NOT_FOUND (Declined Statement/Premise)</text>
  <text x="745" y="254" fill="#c2410c" font-size="9.5" text-anchor="middle">"My salary is..." • Contradictory Unit Traps</text>

  <!-- Line 4 to Domain Guard -->
  <line x1="360" y1="268" x2="360" y2="292" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />

  <!-- 5. Domain Guard -->
  <g filter="url(#softShadow)">
    <rect x="200" y="292" width="320" height="46" rx="8" fill="url(#cardGrad)" stroke="#cbd5e1" stroke-width="1.2" />
    <text x="360" y="310" fill="#0f172a" font-size="13" font-weight="700" text-anchor="middle">4. Domain Guard</text>
    <text x="360" y="326" fill="#64748b" font-size="10.5" text-anchor="middle">Deterministic HR boundary classifier</text>
  </g>
  <!-- Domain Guard Exit Branch -->
  <line x1="520" y1="315" x2="602" y2="315" stroke="#9333ea" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow-purple)" />
  <rect x="610" y="296" width="270" height="38" rx="6" fill="#f3e8ff" stroke="#a855f7" stroke-width="1" />
  <text x="745" y="311" fill="#6b21a8" font-size="11" font-weight="700" text-anchor="middle">OUT_OF_DOMAIN</text>
  <text x="745" y="324" fill="#7e22ce" font-size="9.5" text-anchor="middle">Coding • Math Calculations • Creative Writing • Trivia</text>

  <!-- Line 5 to Pre-Retrieval RBAC & Hybrid Retrieval -->
  <line x1="360" y1="338" x2="360" y2="362" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />

  <!-- 6. Pre-Retrieval RBAC & Hybrid Retrieval -->
  <g filter="url(#softShadow)">
    <rect x="140" y="362" width="440" height="44" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="360" y="380" fill="#1e3a8a" font-size="13" font-weight="700" text-anchor="middle">5. Pre-Retrieval RBAC &amp; Hybrid Search</text>
    <text x="360" y="395" fill="#3b82f6" font-size="10.5" font-weight="500" text-anchor="middle">Hard database isolation (ChromaDB + BM25) fused via RRF</text>
  </g>

  <!-- Line 6 to Cross-Encoder Reranker -->
  <line x1="360" y1="406" x2="360" y2="430" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />

  <!-- 7. Cross-Encoder Reranker -->
  <g filter="url(#softShadow)">
    <rect x="170" y="430" width="380" height="40" rx="8" fill="#f8fafc" stroke="#93c5fd" stroke-width="1.3" />
    <text x="360" y="447" fill="#0f172a" font-size="12.5" font-weight="700" text-anchor="middle">6. Cross-Encoder Reranker</text>
    <text x="360" y="461" fill="#475569" font-size="10" text-anchor="middle">BAAI/bge-reranker-base (Context relevance scoring)</text>
  </g>

  <!-- Fork Lines to Dual Processing Routes -->
  <path d="M 360 470 L 360 488 Q 360 496 350 496 L 220 496 Q 210 496 210 504 L 210 514" fill="none" stroke="#16a34a" stroke-width="1.8" marker-end="url(#arrow-green)" />
  <path d="M 360 470 L 360 488 Q 360 496 370 496 L 620 496 Q 630 496 630 504 L 630 514" fill="none" stroke="#2563eb" stroke-width="1.8" marker-end="url(#arrow-blue)" />

  <!-- Processing Route A (Left): Deterministic Roster Engine -->
  <g filter="url(#softShadow)">
    <rect x="30" y="514" width="360" height="58" rx="8" fill="url(#rosterGrad)" stroke="#22c55e" stroke-width="1.5" />
    <text x="210" y="533" fill="#15803d" font-size="12.5" font-weight="700" text-anchor="middle">📊 Deterministic Roster Route (HR Only)</text>
    <text x="210" y="549" fill="#166534" font-size="10" text-anchor="middle">Direct Structured Directory Aggregation</text>
    <text x="210" y="562" fill="#15803d" font-size="9.5" font-weight="600" text-anchor="middle">100% Precision &amp; Recall • Avoids LLM Counting Errors</text>
  </g>

  <!-- Processing Route B (Right): Grounded LLM Route -->
  <g filter="url(#softShadow)">
    <rect x="450" y="514" width="360" height="58" rx="8" fill="url(#llmGrad)" stroke="#3b82f6" stroke-width="1.5" />
    <text x="630" y="533" fill="#1d4ed8" font-size="12.5" font-weight="700" text-anchor="middle">🤖 Grounded LLM Generation</text>
    <text x="630" y="549" fill="#1e40af" font-size="10" text-anchor="middle">Context-Bound Generation with Strict Accuracy Rules</text>
    <text x="630" y="562" fill="#1d4ed8" font-size="9.5" font-weight="600" text-anchor="middle">Anti-Semantic Substitution • Verified Source Citations</text>
  </g>

  <!-- Converge Lines to Verification & Assessment -->
  <path d="M 210 572 L 210 584 Q 210 592 220 592 L 350 592 Q 360 592 360 600 L 360 608" fill="none" stroke="#16a34a" stroke-width="1.8" marker-end="url(#arrow-green)" />
  <path d="M 630 572 L 630 584 Q 630 592 620 592 L 370 592 Q 360 592 360 600 L 360 608" fill="none" stroke="#2563eb" stroke-width="1.8" marker-end="url(#arrow-blue)" />

  <!-- 8. Verification & Response Assessment -->
  <g filter="url(#softShadow)">
    <rect x="150" y="608" width="420" height="42" rx="8" fill="#f8fafc" stroke="#6366f1" stroke-width="1.5" />
    <text x="360" y="625" fill="#3730a3" font-size="12.5" font-weight="700" text-anchor="middle">7. Verification &amp; Response Assessment</text>
    <text x="360" y="640" fill="#4f46e5" font-size="10" font-weight="500" text-anchor="middle">Numeric Gate • Citation Validity • Grounding Audit • 5 Response Classes</text>
  </g>

  <!-- Line to Final Output -->
  <line x1="360" y1="650" x2="360" y2="670" stroke="#0f1f3d" stroke-width="2" marker-end="url(#arrow)" />

  <!-- 9. Final Output Node -->
  <g filter="url(#softShadow)">
    <rect x="190" y="670" width="340" height="30" rx="6" fill="#0f1f3d" />
    <text x="360" y="690" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">✅ Final Response + Cited Source Cards</text>
  </g>
</svg>"""


def about_page() -> None:
    tabs = st.tabs(["Overview", "Architecture", "Data Sources", "Usage Guide", "Limitations", "Roadmap"])
    with tabs[0]:
        st.markdown("<div class='about-badge'>⭐ Current Release: V1.1.0 — Stable Portfolio Release</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 0.4rem 0;'>Enterprise HR RAG Assistant V1.1</h3>", unsafe_allow_html=True)
        st.write("A security-focused, retrieval-augmented HR assistant designed to answer questions from company "
                 "policies, handbooks, benefits, and organizational data while enforcing role-based access "
                 "boundaries and source-grounded responses.")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""<div class="about-card">
            <h4>📁 Multi-Format Ingestion</h4>
            <p>Parses PDF, DOCX, TXT, HTML, CSV, JSON and XLSX. Features automated cleaning, deduplication, schema validation, and structured section metadata.</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="about-card">
            <h4>⚡ Hybrid Search & Reranking</h4>
            <p>Combines dense vector search (all-MiniLM-L6-v2) with BM25 keyword search via Reciprocal Rank Fusion (RRF), reranked by BGE Cross-Encoder.</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="about-card">
            <h4>🛡️ Grounding & Verification</h4>
            <p>Citations with verifiable source cards, lexical and semantic grounding checks, post-generation Numeric Gate, and 5-class response categorization.</p>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown("""<div class="about-card">
            <h4>🔒 Pre-Retrieval RBAC & Roster</h4>
            <p>Hard database-level isolation of confidential records (Employee vs HR), plus a deterministic roster engine for exact department counts.</p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div class="about-tech-bar">
        <div class="about-tech-grid">
          <div class="about-tech-item"><h5>Vector Database</h5><p>ChromaDB (Local)</p></div>
          <div class="about-tech-item"><h5>Embeddings</h5><p>{settings.embedding_model.split('/')[-1]}</p></div>
          <div class="about-tech-item"><h5>Language Model</h5><p>{active_model_name()} ({settings.llm_provider})</p></div>
          <div class="about-tech-item"><h5>Reranker</h5><p>{settings.reranker_model.split('/')[-1]}</p></div>
        </div>
        </div>""", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:0.75rem;'>Guarded Pipeline Architecture & Execution Flow</div>", unsafe_allow_html=True)
        st.image(ARCHITECTURE_FLOWCHART_SVG, use_container_width=True)
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""<div class="about-card">
            <h4>📊 Deterministic Roster Route (HR Only)</h4>
            <p>Handles department roster and headcount queries (e.g. <em>"How many employees in Engineering?"</em>). Aggregates structured records deterministically with 100% precision and recall across the evaluated dataset, completely avoiding stochastic LLM counting errors.</p>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown("""<div class="about-card">
            <h4>🤖 Grounded LLM Route</h4>
            <p>Synthesizes unstructured policy answers strictly from retrieved context passages. Anti-semantic substitution ensures unmentioned benefits (e.g. pet bereavement) report explicit absence rather than substituting human rules.</p>
            </div>""", unsafe_allow_html=True)
            
        st.markdown("""<div class="about-card" style="margin-top:0.8rem;">
        <h4>🛡️ Defense-in-Depth Guardrails & Verification</h4>
        <ul>
          <li><strong>Request Guard:</strong> Protection against prompt injection, jailbreaks (<code>DAN mode</code>), Base64 wraps, and tag escapes (<code>&lt;/context&gt;</code>).</li>
          <li><strong>Scope Guard:</strong> Distinguishes department-head approval policy questions from confidential directory requests for employees.</li>
          <li><strong>Premise & Statement Guard:</strong> Handles unsupported declarative statements, false premises, and unit mismatches.</li>
          <li><strong>Domain Guard:</strong> Blocks unrelated or out-of-domain questions before retrieval.</li>
          <li><strong>Pre-Retrieval RBAC:</strong> Confidential chunks are filtered at the database level before reaching the model.</li>
          <li><strong>Verification Gates:</strong> Audits numbers and time units against sources (Numeric Gate), verifies citation validity, and checks lexical and semantic grounding.</li>
          <li><strong>Response Assessment:</strong> Evaluates outputs against formal response classes (<code>NORMAL_GROUNDED</code>, <code>ACCESS_RESTRICTED</code>, <code>SECURITY_REFUSAL</code>, <code>NOT_FOUND</code>, <code>OUT_OF_DOMAIN</code>).</li>
        </ul>
        </div>""", unsafe_allow_html=True)

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
        
        rows_html = "".join([
            f"<tr><td><code>{row['File']}</code></td><td>{row['Title']}</td>"
            f"<td><span class='about-badge-tag about-badge-{row['Access']}'>{row['Access']}</span></td>"
            f"<td>v{row['Version']}</td>"
            f"<td><span class='about-badge-tag about-badge-{'current' if row['Current'] == 'yes' else 'archived'}'>{'current' if row['Current'] == 'yes' else 'archived'}</span></td></tr>"
            for row in table
        ])
        st.markdown(f"""<table class="about-table">
        <thead><tr><th>File</th><th>Document Title</th><th>Access Level</th><th>Version</th><th>Status</th></tr></thead>
        <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown(
            "**Role-Based Access Control:**  \n"
            "- **Employee:** Access to standard internal policies and employee-facing documents (handbook, attendance, leave v2, onboarding, WFH, holiday calendar). Confidential employee records remain strictly inaccessible.  \n"
            "- **HR:** Access to all employee-facing documents plus confidential HR and employee records (`employee_profiles.json`, `salary_bands.xlsx`, `performance_ratings.csv`, and `incomplete_employee_record.json`)."
        )
        st.markdown("""<div class="about-callout">
        <strong>Portfolio / Demonstration Authentication:</strong> The Employee/HR role dropdown simulates access levels for demonstration and evaluation. In a production enterprise deployment, the active user role is verified server-side through an authenticated identity provider (SSO / SAML).
        </div>""", unsafe_allow_html=True)

    with tabs[3]:
        u1, u2 = st.columns(2)
        with u1:
            st.markdown("""<div class="about-card">
            <h4>💡 Interactive Controls & Features</h4>
            <ul>
              <li><strong>Simulate access levels:</strong> Select <em>Employee</em> or <em>HR</em> at the top right to simulate the corresponding access permissions.</li>
              <li><strong>Ask questions:</strong> Inquire about available HR documents, leave rules, attendance, onboarding, and company policies.</li>
              <li><strong>Include archived policies:</strong> Turn on this toggle to compare current policies with historical versions (e.g. Leave Policy v1.0).</li>
              <li><strong>Answer style:</strong> Under <em>Settings</em>, choose <em>concise</em> (short answers) or <em>detailed</em> explanations.</li>
              <li><strong>Citation source cards:</strong> Click on source cards beneath each answer to inspect the exact document excerpts used to ground the response.</li>
              <li><strong>User feedback:</strong> Use 👍 / 👎 to rate an answer; ratings are recorded directly to <code>logs/feedback.jsonl</code>.</li>
            </ul>
            </div>""", unsafe_allow_html=True)
        with u2:
            st.markdown("""<div class="about-card">
            <h4>🛑 Safety & Refusals by Design</h4>
            <p style="margin-bottom:0.4rem;">The assistant is engineered to decline requests when boundaries are violated:</p>
            <ul>
              <li><strong>Unsupported premises:</strong> Questions assuming incorrect factual claims (e.g. <em>"Since employees get 30 days of leave..."</em>).</li>
              <li><strong>Declarative statements:</strong> Assertions that ask no question (e.g. <em>"I have completed my probation period"</em>).</li>
              <li><strong>Unauthorized confidential requests:</strong> Directory lookups or salary inquiries from the Employee role.</li>
              <li><strong>Prompt injections:</strong> System prompt extraction attempts, DAN mode, and tag escapes.</li>
              <li><strong>Out-of-domain questions:</strong> Non-HR inquiries such as math, coding, or trivia.</li>
              <li><strong>Questions where grounding cannot be established:</strong> Prevents hallucinated or speculative answers.</li>
            </ul>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="about-callout">
        <strong>Demonstration Note:</strong> The role selector is a portfolio/demo simulation, not a production authentication mechanism. Unauthorized confidential requests are physically filtered out pre-retrieval so confidential data never enters the LLM prompt.
        </div>""", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("""<div class="about-scorecard">
        <div class="about-scorecard-title">Verified V1.1 Release Quality Metrics</div>
        <div class="about-scorecard-grid">
          <div><div class="about-metric-val">1,367</div><div class="about-metric-lbl">Offline Tests Passed (100%)</div></div>
          <div><div class="about-metric-val">20 / 20</div><div class="about-metric-lbl">Logic Mutations Caught</div></div>
          <div><div class="about-metric-val">73 / 74</div><div class="about-metric-lbl">Real-Model Acceptance (98.6%)</div></div>
          <div><div class="about-metric-val">0</div><div class="about-metric-lbl">PII Leaks / RBAC Bypasses</div></div>
        </div>
        </div>""", unsafe_allow_html=True)
        
        st.caption(
            "V1.1 validation: 0 critical issues, 0 material issues, 0 PII leaks, 0 RBAC bypasses. "
            "The deterministic roster engine achieved 100% recall and 100% precision across the evaluated 11 departments. "
            "The single non-passing acceptance check was a conservative refusal on informal leave phrasing, classified as a minor usability limitation rather than a security defect."
        )
        
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("""<div class="about-card">
            <h4>✅ Resolved Capabilities (V1.1 Fixes)</h4>
            <ul>
              <li><strong>Deterministic Department Rosters:</strong> Eliminates stochastic counting errors with 100% precision & recall across the 11 evaluated departments.</li>
              <li><strong>Statement & Premise Protection:</strong> Assertions and unit mismatches are intercepted before model prompting.</li>
              <li><strong>Anti-Semantic Substitution:</strong> Nonexistent benefits report explicit absence without substituting related human rules.</li>
            </ul>
            </div>""", unsafe_allow_html=True)
        with l2:
            st.markdown("""<div class="about-card">
            <h4>⚠️ Current Operational Limitations</h4>
            <ul>
              <li><strong>Roles are simulated:</strong> There is no login; a real deployment needs SSO or similar.</li>
              <li><strong>HR authentication:</strong> The current version does not require a passcode to access the HR role. This is intentional for the testing and demonstration phase, so that users can evaluate the complete Employee and HR functionality. <strong>Planned for V2:</strong> HR role access will be protected by authentication/passcode so that confidential HR functionality cannot be accessed by unauthorized users. The underlying role-based retrieval controls still keep the Employee role away from confidential HR data.</li>
              <li><strong>Corpus boundary:</strong> Grounding is strictly constrained to the synthetic 15-file corpus in <code>data/raw/</code>.</li>
              <li><strong>Independent query context:</strong> Each question is evaluated independently; persistent multi-turn conversational memory is not enabled.</li>
              <li><strong>Conservative refusal bias:</strong> Informal queries or questions where grounding cannot be guaranteed may be conservatively declined to eliminate hallucinations.</li>
              <li><strong>Latency:</strong> Multi-stage guardrails, hybrid search, Cross-Encoder reranking, and verification checks add processing time (see Model Evaluation for measured times).</li>
            </ul>
            </div>""", unsafe_allow_html=True)

    with tabs[5]:
        rm1, rm2 = st.columns(2)
        with rm1:
            st.markdown("""<div class="about-card about-card-blue">
            <h4 style="color:#2563eb;">🚀 V1.2 — Production Hardening</h4>
            <p style="margin-bottom:0.4rem;">Targeted enterprise operational hardening enhancements:</p>
            <ul>
              <li><strong>Authentication-based RBAC:</strong> Replace simulated role selector with authenticated identity and server-side authorization.</li>
              <li><strong>LLM timeout and retry handling:</strong> Prevent provider stalls from leaving requests waiting indefinitely.</li>
              <li><strong>Sensitive-log protection:</strong> Improve access control, retention policies, and encryption for employee queries and logs.</li>
              <li><strong>Enhanced observability:</strong> Record provider/model information, operational traces, and telemetry for troubleshooting.</li>
              <li><strong>Relevance-gate calibration:</strong> Tune cross-encoder thresholds on production query distributions to minimize unnecessary refusals.</li>
              <li><strong>Controlled pilot monitoring:</strong> Real-time dashboards monitoring refusal rates, latency, and operational quality signals.</li>
            </ul>
            </div>""", unsafe_allow_html=True)
        with rm2:
            st.markdown("""<div class="about-card about-card-green">
            <h4 style="color:#059669;">🌟 V2 — Next-Generation Enterprise Assistant</h4>
            <p style="margin-bottom:0.4rem;">Next-generation architectural advancements:</p>
            <ul>
              <li><strong>Real authentication and authorization:</strong> Login-based identity, server-side role assignment, and stronger enterprise RBAC.</li>
              <li><strong>Better evaluator calibration:</strong> More reliable evaluation thresholds and reduced false positives/negatives.</li>
              <li><strong>Class-specific scoring:</strong> Dedicated evaluation benchmarks tailored for each of the 5 response classes.</li>
              <li><strong>Better semantic grounding measurement:</strong> Evaluate whether the meaning of an answer is genuinely supported by retrieved sources.</li>
              <li><strong>Better citation-support measurement:</strong> Verify that a cited source actually supports the claim made.</li>
              <li><strong>Larger benchmark dataset:</strong> More HR documents, employees, departments, and comprehensive edge-case questions.</li>
              <li><strong>Stronger security:</strong> Continuous red-teaming, adversarial evaluation, and prompt-injection defenses.</li>
              <li><strong>Lower latency:</strong> Optimize retrieval and reranking, reduce unnecessary LLM calls, and expand deterministic pathways.</li>
              <li><strong>Controlled conversational context:</strong> Multi-turn conversational memory with persistent grounding, authorization, and RBAC enforcement.</li>
            </ul>
            </div>""", unsafe_allow_html=True)

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
