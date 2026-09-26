"""Central configuration. Values come from environment variables / .env file,
and can be overridden at runtime from the Settings page (saved to data/settings.json)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = Path(os.getenv("RESEARCHMIND_DATA_DIR", ROOT_DIR / "data"))
PAPERS_DIR = DATA_DIR / "papers"
DB_PATH = DATA_DIR / "researchmind.db"
SETTINGS_PATH = DATA_DIR / "settings.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Fallback list used if the live model list can't be fetched from the API.
DEFAULT_CHAT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]
DEFAULT_EMBED_MODELS = ["gemini-embedding-001", "gemini-embedding-2-preview"]


@dataclass
class Settings:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"))
    fallback_model: str = field(default_factory=lambda: os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite"))
    embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"))
    embed_dim: int = int(os.getenv("GEMINI_EMBED_DIM", "768"))
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))
    chunk_size: int = 500  # approx tokens
    chunk_overlap: int = 80
    top_k: int = 8
    output_language: str = "English"

    def save(self) -> None:
        data = asdict(self)
        # Stored locally in data/settings.json (never uploaded anywhere)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        if SETTINGS_PATH.exists():
            try:
                saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                for k, v in saved.items():
                    if hasattr(s, k) and v not in (None, ""):
                        setattr(s, k, v)
            except Exception:
                pass
        # Environment variable always wins for the key if the saved one is empty
        if not s.google_api_key:
            s.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        return s
