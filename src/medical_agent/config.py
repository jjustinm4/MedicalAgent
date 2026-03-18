from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    ollama_fallback_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    critic_confidence_threshold: float = 0.65
    max_retry_loops: int = 2
    chest_xray_model: str = "dima806/chest_xray_pneumonia_detection"
    blip_caption_model: str = "Salesforce/blip-image-captioning-base"
    blip_vqa_model: str = "Salesforce/blip-vqa-base"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def load_settings() -> Settings:
    _load_env_file()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
        ollama_fallback_enabled=_env_bool("OLLAMA_FALLBACK_ENABLED", True),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        critic_confidence_threshold=float(os.getenv("CRITIC_CONFIDENCE_THRESHOLD", "0.65")),
        max_retry_loops=int(os.getenv("MAX_RETRY_LOOPS", "2")),
        chest_xray_model=os.getenv("CHEST_XRAY_MODEL", "dima806/chest_xray_pneumonia_detection"),
        blip_caption_model=os.getenv("BLIP_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"),
        blip_vqa_model=os.getenv("BLIP_VQA_MODEL", "Salesforce/blip-vqa-base"),
    )
