import time
import json
from typing import Optional, Dict, Any, List, Union, Tuple
from threading import Lock

from logging import Logger
from pydantic import Field
import requests
from vllm import LLM, SamplingParams

from src.utils import logger as jsonlogger
from .language_model import LMConfig, LanguageModel, LLMResponseMetrics


class VLLMConfig(LMConfig):
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None  
    gpu_memory_utilization: Optional[float] = None
    trust_remote_code: bool = False
    stop_sequences: Optional[List[str]] = Field(default_factory=lambda: ["FEEDBACK", "OBSERVATION"])
    cache_dir: Optional[str] = None  # Directory to cache/download models
    temperature: Optional[float] = None
    shared_model: bool = False
    
    # Enhanced features for parity with Gemini
    use_chat_template: bool = True  # Use tokenizer.apply_chat_template if available
    json_validation: bool = True    # Validate JSON responses when schema provided
    
    # Optional: use vLLM OpenAI-compatible server instead of in-process engine
    use_server: bool = False
    base_url: Optional[str] = None  # e.g., "http://localhost:8000"
    protocol: str = "openai"       # currently only "openai" supported for server mode
    api_key: Optional[str] = None
    timeout_s: float = 900.0

    # Post-processing options
    strip_think_tags: bool = False  # Remove <think>...</think> from outputs


class VLLMClient(LanguageModel):
    """Synchronous vLLM client compatible with `LanguageModel` interface."""

    _SHARED_ENGINES: Dict[str, Dict[str, Any]] = {}
    _SHARED_LOCK: Lock = Lock()

    def __init__(self, config: VLLMConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        self.logger.info(f"VLLMConfig: {config}")
        self.logger.info(f"use_server: {config.use_server}")
        self.logger.info(f"base_url: {config.base_url}")
        self.logger.info(f"protocol: {config.protocol}")
        self.logger.info(f"api_key: {config.api_key}")
        self.logger.info(f"timeout_s: {config.timeout_s}")
        self._engine: Optional[LLM] = None
        self._engine_lock: Lock = Lock()  # Thread-safe engine initialization
        self._tokenizer = None  # Cache tokenizer for chat templates

    def _shared_engine_key(self) -> str:
        """Generate a cache key for shared engine reuse."""
        cfg = self.config
        payload = {
            "model": cfg.model,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "max_model_len": cfg.max_model_len,
            "dtype": cfg.dtype,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "trust_remote_code": cfg.trust_remote_code,
            "cache_dir": cfg.cache_dir,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _create_engine(self) -> Tuple[LLM, Optional[Any]]:
        """Instantiate a new vLLM engine and tokenizer bundle."""
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.max_model_len is not None:
            kwargs["max_model_len"] = self.config.max_model_len
        if self.config.dtype:
            kwargs["dtype"] = self.config.dtype
        if self.config.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = self.config.gpu_memory_utilization
        if self.config.cache_dir is not None:
            kwargs["download_dir"] = self.config.cache_dir

        engine = LLM(**kwargs)
        tokenizer = None
        if self.config.use_chat_template:
            try:
                tokenizer = engine.get_tokenizer()
            except Exception as e:
                self.logger.warning(f"Failed to get tokenizer for chat templates: {e}")
        return engine, tokenizer

    def _get_shared_engine(self) -> Tuple[LLM, Optional[Any]]:
        """Return a shared engine/tokenizer pair, creating it if needed."""
        key = self._shared_engine_key()
        with self._SHARED_LOCK:
            bundle = self._SHARED_ENGINES.get(key)
            if bundle is None:
                self.logger.info("Creating shared vLLM engine for key=%s", key)
                engine, tokenizer = self._create_engine()
                bundle = {"engine": engine, "tokenizer": tokenizer}
                self._SHARED_ENGINES[key] = bundle
            else:
                self.logger.info("Reusing shared vLLM engine for key=%s", key)
        return bundle["engine"], bundle.get("tokenizer")

    def _init_engine(self) -> LLM:
        """Thread-safe engine initialization (optionally shared)."""
        if self._engine is not None:
            return self._engine

        with self._engine_lock:
            if self._engine is not None:
                return self._engine

            if self.config.use_server:
                raise RuntimeError("Local vLLM engine requested while in server mode")

            if self.config.shared_model:
                engine, tokenizer = self._get_shared_engine()
                self._engine = engine
                self._tokenizer = tokenizer
                return self._engine

            engine, tokenizer = self._create_engine()
            self._engine = engine
            self._tokenizer = tokenizer
            return self._engine

    def _build_prompt(self, messages: List[Dict[str, str]], response_schema: Optional[Dict[str, Any]]) -> str:
        """Build prompt from chat messages, optionally injecting schema instructions.

        Prefers tokenizer.apply_chat_template when available.
        """
        # Prepare messages, optionally appending schema instruction to the last user message
        patched_messages: List[Dict[str, str]] = []
        try:
            if response_schema and self.config.json_validation:
                # Copy messages and append schema instruction to the last user message
                patched_messages = [
                    {"role": m.get("role", ""), "content": str(m.get("content", ""))}
                    for m in messages
                ]
                for i in range(len(patched_messages) - 1, -1, -1):
                    if patched_messages[i].get("role") == "user":
                        patched_messages[i]["content"] = (
                            patched_messages[i].get("content", "")
                            + "\n\nPlease respond with valid JSON matching this schema:\n"
                            + json.dumps(response_schema, indent=2)
                        )
                        break
            else:
                patched_messages = messages
        except Exception as e:
            self.logger.warning(f"Failed to prepare schema-injected messages: {e}")
            patched_messages = messages

        # Try to use chat template if available and enabled
        if self.config.use_chat_template and self._tokenizer is not None:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    patched_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as e:
                self.logger.warning(f"Failed to apply chat template, falling back to simple concatenation: {e}")

        # Fallback: simple concatenation — use last user content with any schema instruction
        system_text = "\n".join([m["content"] for m in patched_messages if m.get("role") == "system"]).strip()
        # Prefer the last user message as the prompt body
        user_text = next((m["content"] for m in reversed(patched_messages) if m.get("role") == "user"), "")
        if system_text:
            return f"{system_text}\n\n{user_text}"
        return user_text

    # -------------------------------
    # OpenAI-compatible server path
    # -------------------------------
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if getattr(self.config, "api_key", None):
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _call_openai_server(self, messages: List[Dict[str, str]], temperature: float, start_time: float) -> Tuple[str, Optional[Dict[str, Any]]]:
        assert self.config.base_url is not None
        url = f"{self.config.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if payload["temperature"] is None:
            del payload["temperature"]
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences

        r = requests.post(url, json=payload, headers=self._headers(), timeout=self.config.timeout_s)
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("No choices in OpenAI-compatible response")
        first = choices[0]
        message = first.get("message") or {}
        text: Optional[str] = message.get("content") or first.get("text")
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
        return text, metrics

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks, tolerant of multiline content.
        If tags are missing/misaligned, return original text.
        """
        try:
            import re
            pattern = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
            return re.sub(pattern, "", text)
        except Exception:
            return text

    def _validate_json_response(self, text: str, response_schema: Optional[Dict[str, Any]]) -> str:
        """Clean JSON response by removing common formatting artifacts."""
        if not response_schema or not self.config.json_validation:
            return text
        
        # Try to extract JSON from response
        cleaned_text = text.strip()
        
        # Validate JSON syntax and return clean JSON
        try:
            parsed_json = json.loads(cleaned_text)
            return json.dumps(parsed_json)  # Return clean, formatted JSON
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}")
            return text  # Return original if parsing fails

    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        call_id = self._begin_call(messages)
        ctx = jsonlogger.json_get_context()
        response_schema = ctx.get("response_schema")

        # Select temperature based on mode
        mode = ctx.get("mode")
        temperature = (
            self.config.val_temperature
            if (mode == "val" or mode == "validation")
            else self.config.train_temperature
        )

        # Ensure tokenizer is ready in local mode so chat templates can be applied
        if not self.config.use_server and self.config.use_chat_template and self._tokenizer is None:
            try:
                self._init_engine()
            except Exception as e:
                self.logger.warning(f"Engine init before prompt build failed: {e}")

        # Build prompt for local engine
        prompt = self._build_prompt(messages, response_schema)

        # Sanitize and validate stop sequences
        stops_cfg = self.config.stop_sequences
        stops: Optional[List[str]] = None
        if isinstance(stops_cfg, list):
            sanitized: List[str] = []
            for s in stops_cfg:
                if isinstance(s, bytes):
                    try:
                        sanitized.append(s.decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                elif isinstance(s, str):
                    sanitized.append(s)
            stops = sanitized if sanitized else None

        # Log request characteristics
        try:
            schema_bytes = len(json.dumps(response_schema)) if response_schema else 0
        except Exception:
            schema_bytes = -1
        self.logger.debug(
            f"vLLM generate: prompt_chars={len(prompt)}, schema_bytes={schema_bytes}, "
            f"stops_count={(len(stops) if stops else 0)}, max_tokens={self.config.max_output_tokens}, "
            f"temperature={temperature}"
        )

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=self.config.max_output_tokens,
            stop=stops,
        )

        start_time = time.time()
        last_err: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 2):
            # Server mode: route to OpenAI-compatible server
            if self.config.use_server and self.config.base_url:
                self.logger.info(f"Calling vLLM server at {self.config.base_url}")
                text, metrics = self._call_openai_server(messages, temperature, start_time)
                # Preserve raw before cleaning
                self._last_raw_output = text
                if response_schema:
                    text = self._validate_json_response(text, response_schema)
                if self.config.strip_think_tags:
                    text = self._strip_think_blocks(text)
                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                return {"text": text, "metrics": metrics}

            # Local engine path
            outputs = self._init_engine().generate([prompt], sampling, use_tqdm=False)
            duration = time.time() - start_time

            if not outputs or len(outputs) == 0:
                raise ValueError("Empty vLLM outputs")

            out0 = outputs[0]
            if not out0.outputs or len(out0.outputs) == 0:
                raise ValueError("vLLM returned no candidates")

            text = out0.outputs[0].text or ""
            # Preserve raw before cleaning
            self._last_raw_output = text

            # Validate and clean JSON response if schema provided
            if response_schema:
                text = self._validate_json_response(text, response_schema)
            if self.config.strip_think_tags:
                text = self._strip_think_blocks(text)

            metrics = self._extract_metrics(out0, duration)

            self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
            return {"text": text, "metrics": metrics}
                
            # except Exception as e:
            #     last_err = e
            #     if attempt > self.config.max_retries:
            #         self.logger.warning(f"Error at attempt {attempt}: Max retries reached, stopping retries")
            #         break
            #     self.logger.warning(f"Warning at attempt {attempt}: Retrying vLLM call: {e}")
            #     delay = min(self.config.starting_delay * (self.config.backoff_factor ** attempt), self.config.max_delay)
            #     time.sleep(delay)

        # On failure, record error with consistent payload structure
        error_payload = {"error": str(last_err) if last_err else "Unknown error"}
        self._end_call(call_id, "", extra=error_payload)
        return {"text": "", "error": error_payload.get("error")}

    def _extract_metrics(self, request_output: Any, duration: float) -> Optional[Dict[str, Any]]:
        """Extract metrics matching GeminiClient's LLMResponseMetrics structure."""

        input_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
        output_tokens = 0
        cand = request_output.outputs[0]
        output_tokens = len(getattr(cand, "token_ids", []) or [])
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        
        return {
            "duration": duration,
            "input_tokens": input_tokens,
            "thinking_tokens": None,  
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
