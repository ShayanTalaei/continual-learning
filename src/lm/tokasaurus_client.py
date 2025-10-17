import os
import time
import json
import base64
from typing import Optional, Dict, Any, List

import requests
import numpy as np
from logging import Logger

from src.lm.language_model import LMConfig, LanguageModel
from src.utils import logger as jsonlogger


class GenerationTruncatedError(RuntimeError):
    """Raised when the model likely hit the max output tokens and truncated."""


class TokasaurusConfig(LMConfig):
    base_url: str
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
            # Request logprobs computed without temperature normalization
            payload["logprobs_ignore_temperature"] = True

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

    def call_with_full_sequence_data(
        self,
        messages: List[Dict[str, str]],
        cartridges: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute chat request and return full sequence tensor data for distillation training.
        
        This method extracts tensor data from the system_fingerprint and combines it with
        input token IDs to provide the complete sequence information needed for training.
        
        Returns:
            Dict containing:
            - text: Generated text
            - metrics: Performance metrics
            - logprobs: OpenAI-style logprobs (if requested)
            - ids: Full sequence token IDs (input + output)
            - topk_logprobs: Top-k logprobs for answer tokens only
            - topk_token_ids: Top-k token IDs for answer tokens only  
            - topk_token_idxs: Indices of answer tokens in full sequence
        """
        call_id = self._begin_call(messages)
        start_time = time.time()
        last_err: Optional[Exception] = None

        # Optional health check
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
                
                # Make the API call with logprobs_in_fingerprint=True to get tensor data
                text, metrics, logprobs, system_fingerprint = self._chat_request_with_fingerprint(
                    messages,
                    start_time,
                    cartridges=cartridges,
                    top_logprobs=top_logprobs,
                )
                
                # Extract tensor data from system_fingerprint
                tensor_data = self._extract_tensor_data_from_fingerprint(
                    system_fingerprint, top_logprobs
                )
                
                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                
                response: Dict[str, Any] = {
                    "text": text,
                    "metrics": metrics,
                    "logprobs": logprobs,
                    **tensor_data
                }
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
                        error_context["response_text"] = getattr(response, 'text', '')[:500]
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
        return {"text": "", "metrics": None, "logprobs": None, "ids": [], "topk_logprobs": None, "topk_token_ids": None, "topk_token_idxs": []}

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        """Tokenize messages to get input token IDs."""
        # This is a simplified version - in practice, you'd want to use the same
        # tokenization logic as the server (apply_chat_template, etc.)
        # For now, we'll extract this from the system_fingerprint or use a basic approach
        
        # Convert messages to a single string for tokenization
        # This is a simplified approach - the actual server uses apply_chat_template
        full_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                full_text += f"System: {content}\n"
            elif role == "user":
                full_text += f"User: {content}\n"
            elif role == "assistant":
                full_text += f"Assistant: {content}\n"
        
        # For now, return empty list - we'll extract input_ids from system_fingerprint
        # In a full implementation, you'd want to use the same tokenizer as the server
        return []

    def _chat_request_with_fingerprint(
        self,
        messages: List[Dict[str, str]],
        start_time: float,
        *,
        cartridges: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
    ) -> "tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]":
        """
        Execute chat request and return (text, metrics, logprobs, system_fingerprint).
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
            "logprobs_in_fingerprint": True,  # Enable tensor data in fingerprint
        }
        if cartridges is not None:
            payload["cartridges"] = cartridges
        if self.cfg.stop_sequences:
            payload["stop"] = self.cfg.stop_sequences
        if top_logprobs is not None:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(top_logprobs)
            payload["logprobs_ignore_temperature"] = True

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

        # Extract system_fingerprint
        system_fingerprint = json.loads(data["system_fingerprint"])

        # Detect truncation
        finish_reason = first.get("finish_reason")
        output_tokens = metrics.get("output_tokens")
        if finish_reason == "length" or (
            isinstance(output_tokens, int) and output_tokens >= self.cfg.max_output_tokens
        ):
            raise GenerationTruncatedError(
                f"Generation likely truncated at max_output_tokens={self.cfg.max_output_tokens} -> Text: {text}"
            )
        return text, metrics, logprobs, system_fingerprint

    def _extract_tensor_data_from_fingerprint(
        self, 
        fingerprint_data: Dict[str, Any], 
        top_logprobs: Optional[int]
    ) -> Dict[str, Any]:
        """Extract tensor data from system_fingerprint and combine with input_ids."""
        
        # Get completion token IDs
        completion_ids = fingerprint_data["completion_ids"][0]  # First sequence
        
        # Initialize tensor data
        tensor_data = {
            "output_ids": completion_ids,
            "topk_logprobs": None,
            "topk_token_ids": None,
        }
        
        # Extract top-k data if available
        packed_topk_logprobs = fingerprint_data["packed_topk_logprobs"]
        packed_topk_indices = fingerprint_data["packed_topk_indices"]
        
        # Decode base64 arrays
        topk_logprobs_bytes = base64.b64decode(packed_topk_logprobs[0])
        topk_indices_bytes = base64.b64decode(packed_topk_indices[0])
        
        # Convert to numpy arrays
        topk_logprobs_array = np.frombuffer(topk_logprobs_bytes, dtype=np.float32)
        topk_indices_array = np.frombuffer(topk_indices_bytes, dtype=np.int32)
        
        # Reshape to [num_tokens, top_k]
        num_tokens = len(completion_ids)
        topk_logprobs_array = topk_logprobs_array.reshape(num_tokens, top_logprobs)
        topk_indices_array = topk_indices_array.reshape(num_tokens, top_logprobs)
        
        tensor_data["topk_logprobs"] = topk_logprobs_array.tolist()
        tensor_data["topk_token_ids"] = topk_indices_array.tolist()
        
        return tensor_data
        


