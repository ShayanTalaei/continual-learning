import os
import time
from typing import Optional, Dict, Any, List

import requests
from logging import Logger

from .language_model import LMConfig, LanguageModel
from src.utils import logger as jsonlogger


class GenerationTruncatedError(RuntimeError):
    """Raised when the model likely hit the max output tokens and truncated."""


class TokasaurusConfig(LMConfig):
    base_url: str = "http://localhost:8080"
    protocol: str = "openai"  # one of {"openai", "toka"}
    api_key: Optional[str] = None
    stop_sequences: Optional[List[str]] = ["FEEDBACK", "OBSERVATION"]
    timeout_s: float = 900.0
    enable_health_check: bool = False  # Ping check before each call (can be noisy under load)


class TokasaurusClient(LanguageModel):
    """Synchronous client for a local Tokasaurus server.

    Supports two protocols:
    - protocol == "openai": expects an OpenAI-compatible API at {base_url}/v1/chat/completions
    - protocol == "toka": best-effort fallback to a minimal native JSON API at {base_url}/generate
    """

    def __init__(self, config: TokasaurusConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        self._http: requests.Session = requests.Session()

    @property
    def cfg(self) -> TokasaurusConfig:  # typed helper
        return self.config  # type: ignore[return-value]

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _ping(self) -> bool:
        try:
            r = self._http.get(f"{self.cfg.base_url}/ping", timeout=5)
            if r.ok:
                data = r.json()
                return data.get("message") == "pong"
            return False
        except Exception:
            return False

    def call(self, system_prompt: str, user_prompt: str) -> str:
        call_id = self._begin_call(system_prompt, user_prompt)
        start_time = time.time()
        last_err: Optional[Exception] = None

        # Optional health check - disabled by default to reduce noise under high load
        if self.cfg.enable_health_check:
            ping_ok = self._ping()
            if not ping_ok:
                self.logger.debug(
                    f"Health check failed for Tokasaurus server at {self.cfg.base_url} "
                    f"(protocol={self.cfg.protocol}, model={self.cfg.model}). "
                    "Proceeding anyway to allow cold start."
                )

        for attempt in range(1, self.config.max_retries + 2):
            try:
                if self.cfg.protocol == "openai":
                    text, metrics = self._call_openai(system_prompt, user_prompt, start_time)
                else:
                    text, metrics = self._call_toka(system_prompt, user_prompt, start_time)
                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                return text
            except Exception as e:
                last_err = e
                error_type = type(e).__name__
                
                # Build detailed error context
                error_context = {
                    "protocol": self.cfg.protocol,
                    "base_url": self.cfg.base_url,
                    "model": self.cfg.model,
                    "error_type": error_type,
                    "system_prompt_len": len(system_prompt),
                    "user_prompt_len": len(user_prompt),
                    "timeout_s": self.cfg.timeout_s,
                }
                
                # Add HTTP-specific details if available
                if hasattr(e, 'response') and e.response is not None:
                    error_context["status_code"] = e.response.status_code
                    try:
                        error_context["response_text"] = e.response.text[:500]  # First 500 chars
                    except:
                        pass
                
                if attempt > self.config.max_retries:
                    self.logger.error(
                        f"Tokasaurus call FAILED after {attempt} attempts. "
                        f"Protocol: {self.cfg.protocol}, URL: {self.cfg.base_url}, "
                        f"Model: {self.cfg.model}, Error: {error_type}: {e}",
                        extra=error_context
                    )
                    break
                
                delay = min(
                    self.config.starting_delay * (self.config.backoff_factor ** attempt),
                    self.config.max_delay,
                )
                self.logger.warning(
                    f"Tokasaurus call failed at attempt {attempt}/{self.config.max_retries + 1}. "
                    f"Protocol: {self.cfg.protocol}, URL: {self.cfg.base_url}, "
                    f"Error: {error_type}: {e}. "
                    f"Retrying in {delay:.2f}s...",
                    extra=error_context
                )
                time.sleep(delay)

        self._end_call(call_id, "", extra={"error": str(last_err) if last_err else "unknown"})
        return ""

    def _call_openai(self, system_prompt: str, user_prompt: str, start_time: float) -> tuple[str, Optional[Dict[str, Any]]]:
        url = f"{self.cfg.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if self.cfg.stop_sequences:
            payload["stop"] = self.cfg.stop_sequences

        r = self._http.post(url, json=payload, headers=self._headers(), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        # Extract content
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("No choices in OpenAI-compatible response")
        first = choices[0]
        message = first.get("message") or {}
        text: Optional[str] = message.get("content")
        if text is None:
            # Some servers return delta or 'text'
            text = first.get("text")
        if text is None:
            raise ValueError("No text content in response")

        duration = time.time() - start_time
        usage = data.get("usage") or {}
        metrics = {
            "duration": duration,
            "input_tokens": usage.get("prompt_tokens"),
            "thinking_tokens": None,
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        # Detect truncation: finish_reason == "length" or output tokens == configured max
        finish_reason = first.get("finish_reason")
        output_tokens = metrics.get("output_tokens")
        if finish_reason == "length" or (
            isinstance(output_tokens, int) and output_tokens >= self.cfg.max_output_tokens
        ):
            raise GenerationTruncatedError(
                f"Generation likely truncated at max_output_tokens={self.cfg.max_output_tokens} -> Text: {text}"
            )
        return text, metrics

    def _call_toka(self, system_prompt: str, user_prompt: str, start_time: float) -> tuple[str, Optional[Dict[str, Any]]]:
        # Minimal JSON API; if your server differs, adjust as needed.
        url = f"{self.cfg.base_url}/generate"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if self.cfg.stop_sequences:
            payload["stop"] = self.cfg.stop_sequences

        r = self._http.post(url, json=payload, headers=self._headers(), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        text: Optional[str] = data.get("text") or data.get("output")
        if text is None:
            raise ValueError("No text field in toka response")

        duration = time.time() - start_time
        usage = data.get("usage") or {}
        metrics = {
            "duration": duration,
            "input_tokens": usage.get("prompt_tokens"),
            "thinking_tokens": None,
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        # Detect truncation for toka API as well
        finish_reason = data.get("finish_reason") or data.get("reason")
        output_tokens = metrics.get("output_tokens")
        if finish_reason == "length" or (
            isinstance(output_tokens, int) and output_tokens >= self.cfg.max_output_tokens
        ):
            raise GenerationTruncatedError(
                f"Generation likely truncated at max_output_tokens={self.cfg.max_output_tokens} -> Text: {text}"
            )
        return text, metrics


