from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import requests

from medical_agent.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class OllamaGemmaClient:
    base_url: str
    model: str
    timeout_seconds: int = 120

    def _generate(self, prompt: str, system: str = "", as_json: bool = False) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if as_json:
            payload["format"] = "json"

        logger.info(
            "Ollama request | model=%s | as_json=%s | prompt_preview=%s",
            self.model,
            as_json,
            prompt.replace("\n", " ")[:160],
        )
        response = requests.post(
            f"{self.base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.exception(
                "Ollama request failed | status=%s | url=%s",
                response.status_code,
                response.url,
            )
            raise
        data = response.json()
        logger.info(
            "Ollama response received | model=%s | response_preview=%s",
            self.model,
            str(data.get("response", "")).replace("\n", " ")[:160],
        )
        return data.get("response", "").strip()

    def generate_text(self, prompt: str, system: str = "") -> str:
        return self._generate(prompt=prompt, system=system, as_json=False)

    def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        raw = self._generate(prompt=prompt, system=system, as_json=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ollama JSON parse failed, attempting object extraction")
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise
