from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

import requests

from medical_agent.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class GeminiClient:
    api_key: str
    model: str
    base_url: str = "https://generativelanguage.googleapis.com"
    timeout_seconds: int = 120
    probe_timeout_seconds: int = 5
    _disabled_reason: str | None = field(default=None, init=False, repr=False)

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _model_resource(self) -> str:
        normalized_model = self.model.strip()
        if normalized_model.startswith("models/"):
            return normalized_model
        return f"models/{normalized_model}"

    def _api_url(self, suffix: str = "") -> str:
        base = self.base_url.rstrip("/")
        resource = self._model_resource()
        return f"{base}/v1beta/{resource}{suffix}"

    def ensure_available(self, force_recheck: bool = False) -> bool:
        if self._disabled_reason and not force_recheck:
            return False

        if not self.api_key.strip():
            self._disabled_reason = "Gemini API key is missing. Set GEMINI_API_KEY in the environment."
            logger.error("Gemini availability check failed | reason=%s", self._disabled_reason)
            return False

        model_url = self._api_url()
        try:
            response = requests.get(
                model_url,
                params={"key": self.api_key},
                timeout=self.probe_timeout_seconds,
            )
        except requests.RequestException as exc:
            self._disabled_reason = (
                "Gemini is not reachable or the API key/model is invalid. "
                "Verify GEMINI_API_KEY and GEMINI_MODEL."
            )
            logger.error("Gemini availability check failed | model=%s | error=%s", self.model, exc)
            return False

        if response.status_code >= 400:
            self._disabled_reason = self._build_http_error_message(response)
            logger.error(
                "Gemini availability check failed | model=%s | message=%s",
                self.model,
                self._disabled_reason,
            )
            return False

        self._disabled_reason = None
        logger.info("Gemini availability check passed | model=%s", self.model)
        return True

    def _build_http_error_message(self, response: requests.Response) -> str:
        body_excerpt = response.text[:240]
        response_error = ""
        try:
            error_payload = response.json().get("error", {})
            response_error = str(error_payload.get("message", "")).strip()
        except ValueError:
            response_error = ""

        if response.status_code == 404:
            return f"Gemini model '{self.model}' was not found."
        if response.status_code == 401:
            return "Gemini authentication failed. Verify GEMINI_API_KEY."
        if response.status_code == 403:
            return "Gemini request was forbidden. Check API key permissions and API enablement."
        if response.status_code == 429:
            return f"Gemini quota exceeded or rate limited: {response_error or body_excerpt}"
        if response_error:
            return f"Gemini request failed ({response.status_code}): {response_error}"

        return f"Gemini request failed ({response.status_code}): {body_excerpt}"

    def _generate(self, prompt: str, system: str = "", as_json: bool = False) -> str:
        if self._disabled_reason:
            raise RuntimeError(self._disabled_reason)

        generation_config: Dict[str, Any] = {
            "temperature": 0.2,
        }
        if as_json:
            generation_config["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        logger.info(
            "Gemini request | model=%s | as_json=%s | prompt_preview=%s",
            self.model,
            as_json,
            prompt.replace("\n", " ")[:160],
        )
        try:
            response = requests.post(
                self._api_url(":generateContent"),
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            self._disabled_reason = (
                "Failed to connect to Gemini. Verify network access and API settings."
            )
            logger.error("Gemini request transport failure | model=%s | error=%s", self.model, exc)
            raise RuntimeError(self._disabled_reason) from None

        if response.status_code >= 400:
            message = self._build_http_error_message(response)
            self._disabled_reason = message
            logger.error(
                "Gemini request failed | status=%s | model=%s | message=%s",
                response.status_code,
                self.model,
                message,
            )
            raise RuntimeError(message)

        self._disabled_reason = None
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise RuntimeError(f"Gemini returned no candidates. Block reason: {block_reason}")
            raise RuntimeError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        logger.info(
            "Gemini response received | model=%s | response_preview=%s",
            self.model,
            text.replace("\n", " ")[:160],
        )
        return text

    def generate_text(self, prompt: str, system: str = "") -> str:
        return self._generate(prompt=prompt, system=system, as_json=False)

    def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        raw = self._generate(prompt=prompt, system=system, as_json=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Gemini JSON parse failed, attempting object extraction")
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout_seconds: int = 120
    probe_timeout_seconds: int = 5
    _disabled_reason: str | None = field(default=None, init=False, repr=False)

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def ensure_available(self, force_recheck: bool = False) -> bool:
        if self._disabled_reason and not force_recheck:
            return False

        tags_url = f"{self.base_url.rstrip('/')}/api/tags"
        try:
            response = requests.get(tags_url, timeout=self.probe_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self._disabled_reason = (
                f"Ollama is not reachable at {self.base_url}. "
                "Ensure Ollama is running and accessible."
            )
            logger.error("Ollama availability check failed | url=%s | error=%s", tags_url, exc)
            return False

        available_models = [str(item.get("name", "")) for item in payload.get("models", [])]
        model_available = any(
            name == self.model
            or name.startswith(f"{self.model}:")
            or self.model.startswith(name.split(":", 1)[0])
            for name in available_models
        )

        if available_models and not model_available:
            self._disabled_reason = (
                f"Model '{self.model}' is not available in Ollama. "
                f"Pull it first with: ollama pull {self.model}"
            )
            logger.error(
                "Ollama model check failed | requested_model=%s | available_models=%s",
                self.model,
                available_models,
            )
            return False

        self._disabled_reason = None
        logger.info("Ollama availability check passed | model=%s | base_url=%s", self.model, self.base_url)
        return True

    def _build_http_error_message(self, response: requests.Response) -> str:
        body_excerpt = response.text[:240]
        response_error = ""
        try:
            response_error = str(response.json().get("error", "")).strip()
        except ValueError:
            response_error = ""

        if response.status_code == 404 and "model" in response_error.lower():
            return (
                f"Ollama model '{self.model}' not found. "
                f"Run: ollama pull {self.model}"
            )
        if response.status_code == 404:
            return (
                f"Ollama endpoint not found at {self.base_url}. "
                "Verify OLLAMA_BASE_URL points to a running Ollama server."
            )
        if response_error:
            return f"Ollama request failed ({response.status_code}): {response_error}"

        return f"Ollama request failed ({response.status_code}): {body_excerpt}"

    def _generate(self, prompt: str, system: str = "", as_json: bool = False) -> str:
        if self._disabled_reason:
            raise RuntimeError(self._disabled_reason)

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
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            self._disabled_reason = (
                f"Failed to connect to Ollama at {self.base_url}. "
                "Ensure the service is running."
            )
            logger.error("Ollama request transport failure | base_url=%s | error=%s", self.base_url, exc)
            raise RuntimeError(self._disabled_reason) from None

        if response.status_code >= 400:
            message = self._build_http_error_message(response)
            self._disabled_reason = message
            logger.error(
                "Ollama request failed | status=%s | message=%s",
                response.status_code,
                message,
            )
            raise RuntimeError(message)

        self._disabled_reason = None
        data = response.json()
        logger.info(
            "Ollama response received | model=%s | response_preview=%s",
            self.model,
            str(data.get("response", "")).replace("\n", " ")[:160],
        )
        return str(data.get("response", "")).strip()

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


@dataclass
class ResilientLLMClient:
    gemini_client: GeminiClient
    ollama_client: OllamaClient | None = None
    prefer_gemini: bool = True
    _disabled_reason: str | None = field(default=None, init=False, repr=False)
    _active_provider: str | None = field(default=None, init=False, repr=False)

    @property
    def active_provider(self) -> str | None:
        return self._active_provider

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _provider_chain(self) -> list[tuple[str, Any]]:
        chain: list[tuple[str, Any]] = []
        if self.prefer_gemini:
            chain.append(("gemini", self.gemini_client))
            if self.ollama_client is not None:
                chain.append(("ollama", self.ollama_client))
        else:
            if self.ollama_client is not None:
                chain.append(("ollama", self.ollama_client))
            chain.append(("gemini", self.gemini_client))
        return chain

    def ensure_available(self, force_recheck: bool = False) -> bool:
        failures: list[str] = []
        for provider_name, provider in self._provider_chain():
            if provider.ensure_available(force_recheck=force_recheck):
                previous_provider = self._active_provider
                self._active_provider = provider_name
                self._disabled_reason = None
                if previous_provider and previous_provider != provider_name:
                    logger.warning(
                        "LLM provider switched | from=%s | to=%s",
                        previous_provider,
                        provider_name,
                    )
                if provider_name == "ollama":
                    logger.warning("Gemini unavailable. Falling back to Ollama provider.")
                return True
            reason = getattr(provider, "disabled_reason", None)
            if reason:
                failures.append(f"{provider_name}: {reason}")

        self._active_provider = None
        self._disabled_reason = " | ".join(failures) if failures else "No configured LLM providers are available."
        return False

    def _call_with_fallback(self, method_name: str, prompt: str, system: str = "") -> Any:
        errors: list[str] = []
        for provider_name, provider in self._provider_chain():
            if not provider.ensure_available(force_recheck=False):
                reason = getattr(provider, "disabled_reason", None)
                if reason:
                    errors.append(f"{provider_name}: {reason}")
                continue

            try:
                result = getattr(provider, method_name)(prompt=prompt, system=system)
                if self._active_provider != provider_name:
                    logger.warning("LLM provider switched | from=%s | to=%s", self._active_provider, provider_name)
                self._active_provider = provider_name
                self._disabled_reason = None
                return result
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                logger.warning("LLM provider call failed | provider=%s | error=%s", provider_name, exc)

        self._active_provider = None
        self._disabled_reason = "All configured LLM providers failed: " + " | ".join(errors)
        raise RuntimeError(self._disabled_reason)

    def generate_text(self, prompt: str, system: str = "") -> str:
        return str(self._call_with_fallback("generate_text", prompt=prompt, system=system))

    def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        data = self._call_with_fallback("generate_json", prompt=prompt, system=system)
        if isinstance(data, dict):
            return data
        raise RuntimeError("LLM provider did not return JSON object.")
