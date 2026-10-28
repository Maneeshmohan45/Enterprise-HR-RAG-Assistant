import os
from unittest.mock import patch, MagicMock
import pytest

from hr_rag.config import Settings, get_api_key, sync_streamlit_secrets, _resolve_llm_provider
from hr_rag.generation import _effective_provider


def test_sync_streamlit_secrets_copies_to_env(monkeypatch):
    """Test that Streamlit Cloud secrets are mirrored into os.environ."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("STREAMLIT_TEST_VAR", raising=False)
    monkeypatch.delenv("STREAM_RESPONSES", raising=False)
    
    mock_st = MagicMock()
    mock_st.secrets = {
        "NVIDIA_API_KEY": "nvapi-mock-secret-key-12345",
        "STREAMLIT_TEST_VAR": "custom_value",
        "STREAM_RESPONSES": True
    }
    
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        sync_streamlit_secrets()
        assert os.environ.get("NVIDIA_API_KEY") == "nvapi-mock-secret-key-12345"
        assert os.environ.get("STREAMLIT_TEST_VAR") == "custom_value"
        assert os.environ.get("STREAM_RESPONSES").lower() == "true"


def test_auto_detect_nvidia_provider(monkeypatch):
    """When only NVIDIA_API_KEY is present and LLM_PROVIDER is unset, auto-detect nvidia."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-mock-key")
    
    provider = _resolve_llm_provider()
    assert provider == "nvidia"
    
    cfg = Settings()
    assert cfg.llm_provider == "nvidia"


def test_auto_detect_claude_provider(monkeypatch):
    """When only ANTHROPIC_API_KEY is present and LLM_PROVIDER is unset, auto-detect claude."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-mock-key")
    
    provider = _resolve_llm_provider()
    assert provider == "claude"
    
    cfg = Settings()
    assert cfg.llm_provider == "claude"


def test_explicit_llm_provider_takes_precedence(monkeypatch):
    """Explicit LLM_PROVIDER setting in env wins even if another key is present."""
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-mock-key")
    
    provider = _resolve_llm_provider()
    assert provider == "claude"
    
    cfg = Settings(llm_provider="nvidia")
    assert cfg.llm_provider == "nvidia"


def test_get_api_key_success_and_missing_error(monkeypatch):
    """Test get_api_key returns the key or raises clear error when missing."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-mock-123")
    assert get_api_key("nvidia") == "nvapi-mock-123"
    
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is missing"):
        get_api_key("nvidia")


def test_generation_effective_provider_fallback(monkeypatch):
    """If default config was claude but user only supplied NVIDIA key, generation routes to nvidia."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-mock-key")
    
    cfg = Settings(llm_provider="claude")
    assert _effective_provider(cfg) == "nvidia"
