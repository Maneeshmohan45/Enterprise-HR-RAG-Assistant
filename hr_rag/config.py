"""Central configuration.

Everything that might change between machines or experiments lives here.
Secrets (API keys) are read from the environment / a local ``.env`` file and
are never written in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reads PROJECT_ROOT/.env if it exists. Real environment variables win over it.
load_dotenv(PROJECT_ROOT / ".env")


def sync_streamlit_secrets() -> None:
    """If running inside Streamlit with secrets configured, copy them to os.environ.

    Preserves any existing environment variables. Catches StreamlitSecretNotFoundError.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                    os.environ[k] = str(v)
    except Exception:
        pass


sync_streamlit_secrets()


def _resolve_llm_provider() -> str:
    """Return the configured LLM provider, auto-detecting based on available keys if not explicit."""
    sync_streamlit_secrets()
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("claude", "nvidia"):
        return explicit
    has_nvidia = bool(os.getenv("NVIDIA_API_KEY", "").strip())
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    if has_nvidia and not has_claude:
        return "nvidia"
    if has_claude and not has_nvidia:
        return "claude"
    return "claude"


def _optional_float(name: str, default: float | None) -> float | None:
    """Read a float from the environment. An empty value means 'not set'."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return float(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    # ---- paths -------------------------------------------------------------
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.yaml"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    logs_dir: Path = PROJECT_ROOT / "logs"
    collection_name: str = "hr_knowledge_base"

    # ---- supported file types ----------------------------------------------
    supported_extensions: tuple[str, ...] = (
        ".pdf", ".docx", ".txt", ".csv", ".json", ".xlsx", ".html",
    )

    # ---- chunking ----------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- models ------------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )

    # ---- retrieval ---------------------------------------------------------
    top_k: int = 8                  # candidates per retriever (vector and BM25)
    rrf_k: int = 60                 # Reciprocal Rank Fusion constant
    rrf_max_results: int = 18       # candidates sent to the reranker
    rerank_top_n: int = 14          # kept after reranking
    # Simple questions get a small prompt (cheaper, faster). Multi-part, "list all" and
    # "compare" questions get the larger one.
    context_chunks: int = 6
    max_context_chars: int = 7000
    context_chunks_complex: int = 12
    max_context_chars_complex: int = 14000
    max_query_chars: int = 1000

    # ---- LLM ---------------------------------------------------------------
    llm_provider: str = field(
        default_factory=_resolve_llm_provider
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    )
    nvidia_model: str = field(
        default_factory=lambda: os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    )
    nvidia_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )
    # "false" turns thinking off for NVIDIA reasoning models, "true" turns it on,
    # empty = do not send the option (needed for models that do not support it).
    nvidia_enable_thinking: str = field(
        default_factory=lambda: os.getenv("NVIDIA_ENABLE_THINKING", "").strip().lower()
    )
    llm_temperature: float | None = field(
        default_factory=lambda: _optional_float("LLM_TEMPERATURE", 0.2)
    )
    # "concise" = short, simple answers (default). "detailed" = fuller explanations.
    answer_style: str = field(
        default_factory=lambda: os.getenv("ANSWER_STYLE", "concise").strip().lower()
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048") or 2048)
    )

    # ---- request guard (prompt injection / role override) -------------------
    # "enforce" = decline the whole request, "shadow" = answer as before but record what would have
    # been declined, "off" = disabled. This only ever narrows what the assistant does; it never
    # touches the role filter (Phase 1).
    injection_guard_mode: str = field(
        default_factory=lambda: os.getenv("INJECTION_GUARD_MODE", "enforce").strip().lower()
    )

    # ---- out-of-domain handling ---------------------------------------------
    # Domain rules (arithmetic, code, creative writing, general knowledge) are deterministic: enforce.
    domain_guard_mode: str = field(
        default_factory=lambda: os.getenv("DOMAIN_GUARD_MODE", "enforce").strip().lower()
    )
    # Relevance gate: a second, isolated layer. It compares the best reranker score with a threshold
    # that must be calibrated on the real reranker (scripts/calibrate_relevance.py). Until then it only
    # records what it would have done ("shadow"); "enforce" without a threshold is refused at start.
    relevance_gate_mode: str = field(
        default_factory=lambda: os.getenv("RELEVANCE_GATE_MODE", "shadow").strip().lower()
    )
    relevance_min_rerank_score: float | None = field(
        default_factory=lambda: (float(os.getenv("RELEVANCE_MIN_RERANK_SCORE"))
                                 if os.getenv("RELEVANCE_MIN_RERANK_SCORE", "").strip() else None)
    )
    # Numeric gate: an answer may only state numbers (and time units) that the sources give or the question
    # already contained. It withholds invented, converted or calculated figures. Deterministic: enforce.
    numeric_gate_mode: str = field(
        default_factory=lambda: os.getenv("NUMERIC_GATE_MODE", "enforce").strip().lower()
    )
    # Premise guard: user-stated claims are not evidence. "enforce" declines a statement that asks nothing and a
    # question whose figure the documents contradict; "shadow" only records; "off" disables it.
    premise_guard_mode: str = field(
        default_factory=lambda: os.getenv("PREMISE_GUARD_MODE", "enforce").strip().lower()
    )
    # Roster lookup: a pure roster or headcount question is answered from the department's complete roster chunk
    # (complete by construction) instead of by the model. "shadow" only records; "off" disables it.
    roster_lookup_mode: str = field(
        default_factory=lambda: os.getenv("ROSTER_LOOKUP_MODE", "enforce").strip().lower()
    )
    # Show the answer as it is generated (the chat updates while the model writes).
    stream_responses: bool = field(
        default_factory=lambda: os.getenv("STREAM_RESPONSES", "true").strip().lower() not in ("0", "false", "no", "off")
    )

    # ---- evaluation thresholds (teaching values; tune on real data) --------
    lexical_grounding_threshold: float = 0.45
    semantic_grounding_threshold: float = 0.50
    answer_relevance_threshold: float = 0.45


# Which document classifications each role may retrieve.
ROLE_ACCESS: dict[str, list[str]] = {
    "employee": ["public", "internal"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "confidential"],
    "admin": ["public", "internal", "confidential", "restricted"],
}

settings = Settings()


def get_api_key(provider: str | None = None) -> str:
    """Return the API key for a provider or raise a clear error."""
    sync_streamlit_secrets()
    if provider is None:
        provider = _resolve_llm_provider()
    env_name = {"claude": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY"}.get(provider)
    if env_name is None:
        raise ValueError(f"Unsupported LLM_PROVIDER='{provider}'. Use 'claude' or 'nvidia'.")
    key = os.getenv(env_name, "").strip()
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and env_name in st.secrets and str(st.secrets[env_name]).strip():
                key = str(st.secrets[env_name]).strip()
                os.environ[env_name] = key
        except Exception:
            pass
    if not key:
        raise ValueError(
            f"{env_name} is missing. Copy .env.example to .env and add your own key."
        )
    return key