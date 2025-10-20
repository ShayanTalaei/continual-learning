import time
import json
from typing import Optional, Dict, Any, List, Union
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
    strip_code_fences: bool = True  # Remove ```...``` markdown fences and extract inner code


class VLLMClient(LanguageModel):
    """Synchronous vLLM client compatible with `LanguageModel` interface."""

    def __init__(self, config: VLLMConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        logger.info(f"VLLMConfig: {config}")
        logger.info(f"use_server: {config.use_server}")
        logger.info(f"base_url: {config.base_url}")
        logger.info(f"protocol: {config.protocol}")
        logger.info(f"api_key: {config.api_key}")
        logger.info(f"timeout_s: {config.timeout_s}")
        self._engine: Optional[LLM] = None
        self._engine_lock: Lock = Lock()  # Thread-safe engine initialization
        self._tokenizer = None  # Cache tokenizer for chat templates

    def _init_engine(self) -> LLM:
        """Thread-safe engine initialization."""
        if self._engine is not None:
            return self._engine
        
        with self._engine_lock:
            # Double-check pattern for thread safety
            if self._engine is not None:
                return self._engine
                
            model_id = self.config.model

            kwargs: Dict[str, Any] = {
                "model": model_id,
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

            self._engine = LLM(**kwargs)
            
            # Cache tokenizer for chat templates if enabled
            if self.config.use_chat_template:
                try:
                    self._tokenizer = self._engine.get_tokenizer()
                except Exception as e:
                    self.logger.warning(f"Failed to get tokenizer for chat templates: {e}")
                    self._tokenizer = None
            
            return self._engine

    def _build_prompt(self, system_prompt: str, user_prompt: str, response_schema: Optional[Dict[str, Any]]) -> str:
        """Build prompt with chat templates and schema-based instructions."""
        # Try to use chat template if available and enabled
        if self.config.use_chat_template and self._tokenizer is not None:
            try:
                # Build messages for chat template
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # Add schema information to user message if present
                if response_schema and self.config.json_validation:
                    schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(response_schema, indent=2)}"
                    user_prompt_with_schema = user_prompt + schema_instruction
                else:
                    user_prompt_with_schema = user_prompt
                
                messages.append({"role": "user", "content": user_prompt_with_schema})
                
                # Apply chat template
                prompt = self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                return prompt
            except Exception as e:
                self.logger.warning(f"Failed to apply chat template, falling back to simple concatenation: {e}")
        
        # Fallback: simple concatenation with schema information
        if response_schema and self.config.json_validation:
            schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(response_schema, indent=2)}"
            return f"{system_prompt}\n\n{user_prompt}{schema_instruction}"
        
        return f"{system_prompt}\n\n{user_prompt}"

    # -------------------------------
    # OpenAI-compatible server path
    # -------------------------------
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if getattr(self.config, "api_key", None):
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _call_openai_server(self, system_prompt: str, user_prompt: str, start_time: float) -> tuple[str, Optional[Dict[str, Any]]]:
        assert self.config.base_url is not None
        url = f"{self.config.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
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

    def _strip_code_fences(self, text: str) -> str:
        """Extract inner code from triple-backtick blocks; if none, return original.
        Handles optional language tag and multiline content.
        """
        try:
            import re
            pattern = re.compile(r"```[^\n]*\n([\s\S]*?)\n```", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                return m.group(1).strip()
            # Also handle single-line fenced variants without trailing newline before ```
            pattern2 = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.IGNORECASE)
            m2 = pattern2.search(text)
            if m2:
                return m2.group(1).strip()
            return text
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

    def call(self, system_prompt: str, user_prompt: str) -> str:
        call_id = self._begin_call(system_prompt, user_prompt)
        ctx = jsonlogger.json_get_context()
        response_schema = ctx.get("response_schema")
        # Ensure tokenizer is ready in local mode so chat templates can be applied
        if not self.config.use_server and self.config.use_chat_template and self._tokenizer is None:
            try:
                self._init_engine()
            except Exception as e:
                self.logger.warning(f"Engine init before prompt build failed: {e}")

        prompt = self._build_prompt(system_prompt, user_prompt, response_schema)
        print(f"Prompt: {prompt}")
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

        # Log request characteristics to aid debugging oversized/malformed inputs
        try:
            schema_bytes = len(json.dumps(response_schema)) if response_schema else 0
        except Exception:
            schema_bytes = -1
        self.logger.debug(
            f"vLLM generate: prompt_chars={len(prompt)}, schema_bytes={schema_bytes}, "
            f"stops_count={(len(stops) if stops else 0)}, max_tokens={self.config.max_output_tokens}, "
            f"temperature={self.config.temperature}"
        )

        sampling = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            stop=stops,
        )

        start_time = time.time()
        last_err: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 2):
            # try:
                # If server mode enabled, route to OpenAI-compatible server
            self.logger.info(f"use_server: {self.config.use_server}, base_url: {self.config.base_url}")
            if self.config.use_server and self.config.base_url:
                self.logger.info(f"Calling vLLM server at {self.config.base_url}")
                text, metrics = self._call_openai_server(system_prompt, user_prompt, start_time)
                if response_schema:
                    text = self._validate_json_response(text, response_schema)
                if self.config.strip_code_fences:
                    text = self._strip_code_fences(text)
                if self.config.strip_think_tags:
                    text = self._strip_think_blocks(text)
                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                return text

            outputs = self._init_engine().generate([prompt], sampling, use_tqdm=False)
            duration = time.time() - start_time

            if not outputs or len(outputs) == 0:
                raise ValueError("Empty vLLM outputs")

            out0 = outputs[0]
            # Prefer first candidate text
            if not out0.outputs or len(out0.outputs) == 0:
                raise ValueError("vLLM returned no candidates")

            text = out0.outputs[0].text or ""
            
            # Validate and clean JSON response if schema provided
            if response_schema:
                text = self._validate_json_response(text, response_schema)
            if self.config.strip_code_fences:
                text = self._strip_code_fences(text)
            if self.config.strip_think_tags:
                text = self._strip_think_blocks(text)
            
            metrics = self._extract_metrics(out0, duration)

            self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
            return text
                
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
        return ""

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
