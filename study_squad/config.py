"""Central configuration.

Values are read in this order: environment variables / ``.env`` file, then
Streamlit secrets (``.streamlit/secrets.toml`` locally or the Secrets panel on
Streamlit Community Cloud), then the defaults below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Groq retired llama-3.3-70b-versatile for free/dev tiers on 16 Aug 2026 and
# recommends openai/gpt-oss-120b as the replacement.
DEFAULT_MODEL = "openai/gpt-oss-120b"
AVAILABLE_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

# Every model below is served by Groq and called with your Groq API key. The
# prefix in the ID is who *trained* the open-weight model, not whose API is used.
MODEL_LABELS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B · best quality",
    "openai/gpt-oss-20b": "GPT-OSS 20B · fastest",
    "qwen/qwen3.8-27b": "Qwen 3.8 27B · alternative",
}


def model_label(model_id: str) -> str:
    """Friendly name for the model picker, e.g. 'GPT-OSS 120B · best quality'."""
    return MODEL_LABELS.get(model_id, model_id.split("/")[-1])


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return a setting from the environment, then Streamlit secrets, then ``default``."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file, or Streamlit not running
        pass
    return default


PLACEHOLDER_KEYS = {"your_groq_api_key_here", "gsk_your_real_key", "gsk_...", "gsk_…"}


def clean_api_key(value: str | None) -> str | None:
    """Strip spaces, line breaks and stray quotes that often sneak in when pasting a key.

    Returns None for empty values and for the placeholder text from the templates.
    """
    if value is None:
        return None
    key = str(value).strip().strip("\"'").strip()
    if not key or key in PLACEHOLDER_KEYS:
        return None
    return key


def key_problem(raw: str | None) -> str | None:
    """Explain what looks wrong with a configured key, or None if it looks fine."""
    if raw is None or not str(raw).strip():
        return None
    key = clean_api_key(raw)
    if key is None:
        return "The Groq key is still the placeholder text from the template."
    if not key.startswith("gsk_"):
        return "The Groq key should start with gsk_. It may have been pasted incompletely."
    if len(key) < 40:
        return "The Groq key looks too short. It may have been cut off when pasting."
    return None


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_key_problem: str | None
    model: str
    reasoning_effort: str
    temperature: float
    history_turns: int
    data_dir: Path
    persist_logs: bool

    @property
    def log_path(self) -> Path:
        return self.data_dir / "study_logs.jsonl"

    @property
    def feedback_path(self) -> Path:
        return self.data_dir / "feedback.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def sample_log_path(self) -> Path:
        return PROJECT_ROOT / "data" / "sample_logs.jsonl"


def load_settings() -> Settings:
    data_dir = Path(get_setting("STUDY_SQUAD_DATA_DIR", str(PROJECT_ROOT / "data")))
    return Settings(
        groq_api_key=clean_api_key(raw_key := get_setting("GROQ_API_KEY")),
        groq_key_problem=key_problem(raw_key),
        model=get_setting("GROQ_MODEL", DEFAULT_MODEL),
        reasoning_effort=get_setting("GROQ_REASONING_EFFORT", "low"),
        temperature=float(get_setting("STUDY_SQUAD_TEMPERATURE", "0.6")),
        history_turns=int(get_setting("STUDY_SQUAD_HISTORY_TURNS", "3")),
        data_dir=data_dir,
        persist_logs=_as_bool(get_setting("STUDY_SQUAD_PERSIST_LOGS"), default=True),
    )
