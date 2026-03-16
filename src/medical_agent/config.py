from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma:2b"
    critic_confidence_threshold: float = 0.65
    max_retry_loops: int = 2
    blip_caption_model: str = "Salesforce/blip-image-captioning-base"
    blip_vqa_model: str = "Salesforce/blip-vqa-base"


def _load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def load_settings() -> Settings:
    _load_env_file()
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma:2b"),
        critic_confidence_threshold=float(os.getenv("CRITIC_CONFIDENCE_THRESHOLD", "0.65")),
        max_retry_loops=int(os.getenv("MAX_RETRY_LOOPS", "2")),
        blip_caption_model=os.getenv("BLIP_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"),
        blip_vqa_model=os.getenv("BLIP_VQA_MODEL", "Salesforce/blip-vqa-base"),
    )
