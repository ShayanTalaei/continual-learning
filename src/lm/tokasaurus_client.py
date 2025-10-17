import os
import time
from typing import Optional, Dict, Any, List

import requests
from logging import Logger

from src.lm.language_model import LMConfig, LanguageModel
from src.utils import logger as jsonlogger


class GenerationTruncatedError(RuntimeError):
    """Raised when the model likely hit the max output tokens and truncated."""


class TokasaurusConfig(LMConfig):
    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    stop_sequences: Optional[List[str]] = ["FEEDBACK", "OBSERVATION"]
    timeout_s: float = 900.0
    enable_health_check: bool = False  # Ping check before each call (can be noisy under load)


class TokasaurusClient(LanguageModel):
    """Synchronous client for a local Tokasaurus server.

    Expects an OpenAI-compatible API:
    - Chat completions at {base_url}/v1/chat/completions
    - Cartridge chat completions at {base_url}/v1/cartridge/chat/completions
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

    def call(
        self,
        messages: List[Dict[str, str]],
        cartridges: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        call_id = self._begin_call(messages)
        start_time = time.time()
        last_err: Optional[Exception] = None

        # Optional health check - disabled by default to reduce noise under high load
        if self.cfg.enable_health_check:
            ping_ok = self._ping()
            if not ping_ok:
                self.logger.debug(
                    f"Health check failed for Tokasaurus server at {self.cfg.base_url} "
                    f"(model={self.cfg.model}). "
                    "Proceeding anyway to allow cold start."
                )

        for attempt in range(1, self.config.max_retries + 2):
            try:
                text, metrics, logprobs = self._chat_request(
                    messages,
                    start_time,
                    cartridges=cartridges,
                    top_logprobs=top_logprobs,
                )
                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                response: Dict[str, Any] = {"text": text}
                if metrics:
                    response["metrics"] = metrics
                if logprobs:
                    response["logprobs"] = logprobs
                return response
            except Exception as e:
                last_err = e
                error_type = type(e).__name__
                
                # Build detailed error context
                error_context = {
                    "base_url": self.cfg.base_url,
                    "model": self.cfg.model,
                    "error_type": error_type,
                    "messages_count": len(messages),
                    "timeout_s": self.cfg.timeout_s,
                }
                
                # Add HTTP-specific details if available
                if hasattr(e, 'response') and getattr(e, 'response', None) is not None:
                    response = getattr(e, 'response')
                    error_context["status_code"] = getattr(response, 'status_code', None)
                    try:
                        error_context["response_text"] = getattr(response, 'text', '')[:500]  # First 500 chars
                    except:
                        pass
                
                if attempt > self.config.max_retries:
                    self.logger.error(
                        f"Tokasaurus call FAILED after {attempt} attempts. "
                        f"URL: {self.cfg.base_url}, Model: {self.cfg.model}, Error: {error_type}: {e}",
                        extra=error_context
                    )
                    break
                
                delay = min(
                    self.config.starting_delay * (self.config.backoff_factor ** attempt),
                    self.config.max_delay,
                )
                self.logger.warning(
                    f"Tokasaurus call failed at attempt {attempt}/{self.config.max_retries + 1}. "
                    f"URL: {self.cfg.base_url}, Error: {error_type}: {e}. "
                    f"Retrying in {delay:.2f}s...",
                    extra=error_context
                )
                time.sleep(delay)

        self._end_call(call_id, "", extra={"error": str(last_err) if last_err else "unknown"})
        return {"text": ""}

    def _chat_request(
        self,
        messages: List[Dict[str, str]],
        start_time: float,
        *,
        cartridges: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
    ) -> "tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]":
        """
        Execute chat request and return (text, metrics, logprobs).
        
        Returns:
            text: Generated text
            metrics: Performance metrics (duration, token counts)
            logprobs: Logprobs data (only when top_logprobs is requested)
        """
        # Select endpoint based on whether cartridges are provided
        if cartridges is None:
            url = f"{self.cfg.base_url}/v1/chat/completions"
        else:
            url = f"{self.cfg.base_url}/v1/cartridge/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if cartridges is not None:
            payload["cartridges"] = cartridges
        if self.cfg.stop_sequences:
            payload["stop"] = self.cfg.stop_sequences
        if top_logprobs is not None:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(top_logprobs)

        r = self._http.post(url, json=payload, headers=self._headers(), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        # Extract content
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("No choices in response")
        first = choices[0]
        message = first.get("message") or {}
        text: Optional[str] = message.get("content")
        if text is None:
            # Some servers return 'text'
            text = first.get("text")
        if text is None:
            raise ValueError("No text content in response")

        duration = time.time() - start_time
        usage = data.get("usage") or {}
        metrics: Dict[str, Any] = {
            "duration": duration,
            "input_tokens": usage.get("prompt_tokens"),
            "thinking_tokens": None,
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        
        # Extract logprobs if requested
        logprobs: Optional[Dict[str, Any]] = None
        if top_logprobs is not None:
            logprobs_data = first.get("logprobs")
            if logprobs_data is not None:
                logprobs = logprobs_data

        # Detect truncation: finish_reason == "length" or output tokens == configured max
        finish_reason = first.get("finish_reason")
        output_tokens = metrics.get("output_tokens")
        if finish_reason == "length" or (
            isinstance(output_tokens, int) and output_tokens >= self.cfg.max_output_tokens
        ):
            raise GenerationTruncatedError(
                f"Generation likely truncated at max_output_tokens={self.cfg.max_output_tokens} -> Text: {text}"
            )
        return text, metrics, logprobs


