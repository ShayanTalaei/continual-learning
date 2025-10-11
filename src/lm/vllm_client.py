import time
import json
from typing import Optional, Dict, Any, List, Union
from threading import Lock

from logging import Logger
from pydantic import Field
from vllm import LLM, SamplingParams

from src.utils import logger as jsonlogger
from .language_model import LMConfig, LanguageModel, LLMResponseMetrics


class VLLMConfig(LMConfig):
    # model: inherited from LMConfig; use any HuggingFace model ID or local path
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None  # e.g., "float16", "bfloat16"
    gpu_memory_utilization: Optional[float] = None
    trust_remote_code: bool = False
    stop_sequences: Optional[List[str]] = Field(default_factory=lambda: ["FEEDBACK", "OBSERVATION"])
    
    # Enhanced features for parity with Gemini
    use_chat_template: bool = True  # Use tokenizer.apply_chat_template if available
    json_validation: bool = True    # Validate JSON responses when schema provided
    rate_limit_delay: float = 0.1   # Sleep between retries to emulate throttling


class VLLMClient(LanguageModel):
    """Synchronous vLLM client compatible with `LanguageModel` interface."""

    def __init__(self, config: VLLMConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
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
                
            # Strip "vllm:" prefix if present (factory routes based on "vllm" substring)
            model_id = self.config.model
            if model_id.startswith("vllm:"):
                model_id = model_id[len("vllm:"):]

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
        """Build prompt with enhanced features: chat templates, JSON enforcement."""
        
        # Try to use chat template if available and enabled
        if self.config.use_chat_template and self._tokenizer is not None:
            try:
                # Build messages for chat template
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                
                # Apply chat template
                prompt = self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Add JSON enforcement if schema provided
                if response_schema and self.config.json_validation:
                    json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. Do not include any text before or after the JSON."
                    prompt = prompt.replace("<|im_start|>assistant\n", f"<|im_start|>assistant{json_instruction}\n")
                
                return prompt
            except Exception as e:
                self.logger.warning(f"Failed to apply chat template, falling back to simple concatenation: {e}")
        
        # Fallback: simple concatenation with JSON enforcement
        if response_schema and self.config.json_validation:
            json_instruction = "IMPORTANT: Respond ONLY with valid JSON matching the provided schema. Do not include any text before or after the JSON."
            return f"{system_prompt}\n\n{json_instruction}\n\n{user_prompt}"
        
        return f"{system_prompt}\n\n{user_prompt}"

    def _validate_json_response(self, text: str, response_schema: Optional[Dict[str, Any]]) -> str:
        """Clean and validate JSON response using standard library only."""
        if not response_schema or not self.config.json_validation:
            return text
        
        # Try to extract JSON from response
        cleaned_text = text.strip()
        
        # Remove common prefixes/suffixes
        for prefix in ["```json", "```", "Response:", "Answer:"]:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        for suffix in ["```", "Response:", "Answer:"]:
            if cleaned_text.endswith(suffix):
                cleaned_text = cleaned_text[:-len(suffix)].strip()
        
        # Validate JSON syntax only (no schema validation to match GeminiClient behavior)
        try:
            parsed_json = json.loads(cleaned_text)
            return json.dumps(parsed_json)  # Return clean JSON
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON validation failed: {e}")
            return text  # Return original if validation fails

    def call(self, system_prompt: str, user_prompt: str) -> str:
        call_id = self._begin_call(system_prompt, user_prompt)
        ctx = jsonlogger.json_get_context()
        response_schema = ctx.get("response_schema")

        sampling = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            stop=self.config.stop_sequences or None,
        )

        prompt = self._build_prompt(system_prompt, user_prompt, response_schema)

        start_time = time.time()
        last_err: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                # Add rate limiting delay between retries
                if attempt > 1 and self.config.rate_limit_delay > 0:
                    time.sleep(self.config.rate_limit_delay)
                
                outputs = self._init_engine().generate([prompt], sampling)
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
                
                metrics = self._extract_metrics(out0, duration)

                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                return text
                
            except Exception as e:
                last_err = e
                if attempt > self.config.max_retries:
                    self.logger.warning(f"Error at attempt {attempt}: Max retries reached, stopping retries")
                    break
                self.logger.warning(f"Warning at attempt {attempt}: Retrying vLLM call: {e}")
                delay = min(self.config.starting_delay * (self.config.backoff_factor ** attempt), self.config.max_delay)
                time.sleep(delay)

        # On failure, record error with consistent payload structure
        error_payload = {"error": str(last_err) if last_err else "Unknown error"}
        self._end_call(call_id, "", extra=error_payload)
        return ""

    def _extract_metrics(self, request_output: Any, duration: float) -> Optional[Dict[str, Any]]:
        """Extract metrics matching GeminiClient's LLMResponseMetrics structure."""
        # vLLM RequestOutput: prompt_token_ids, outputs[0].token_ids
        input_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
        output_tokens = 0
        cand = request_output.outputs[0]
        output_tokens = len(getattr(cand, "token_ids", []) or [])
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        
        # Match GeminiClient's LLMResponseMetrics structure exactly
        return {
            "duration": duration,
            "input_tokens": input_tokens,
            "thinking_tokens": None,  # vLLM doesn't have thinking tokens
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
