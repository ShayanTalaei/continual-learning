import os
import time
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerationConfig
from google.oauth2 import service_account
from logging import Logger
from src.utils import logger as jsonlogger

from .language_model import LMConfig, LanguageModel

load_dotenv(override=True)

class GeminiConfig(LMConfig):
    thinking_budget: Optional[int] = None
    stop_sequences: Optional[List[str]] = ["FEEDBACK", "OBSERVATION"]

class GeminiClient(LanguageModel):
    """Synchronous Gemini client compatible with `LanguageModel` interface."""

    def __init__(self, config: GeminiConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        self._gemini_client: Optional[genai.Client] = None

    @property
    def cfg(self) -> GeminiConfig:  # typed helper
        return self.config  # type: ignore[return-value]

    def call(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        call_id = self._begin_call(messages)
        ctx = jsonlogger.json_get_context()
        response_schema = ctx.get("response_schema")
        use_json_mode = response_schema is not None
        
        # Separate system messages from conversation messages
        system_instruction = None
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                if system_instruction is None:
                    system_instruction = msg["content"]
                else:
                    # If multiple system messages, concatenate them
                    system_instruction += "\n" + msg["content"]
            else:
                # Convert to Gemini format: {"role": "user"|"model", "parts": [{"text": content}]}
                role = "user" if msg["role"] == "user" else "model"  # Gemini uses "model" instead of "assistant"
                conversation_messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        
        generate_content_config = GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.config.temperature ,
            max_output_tokens=self.config.max_output_tokens,
            response_mime_type=("application/json" if use_json_mode else None),
            thinking_config=(
                ThinkingConfig(thinking_budget=self.cfg.thinking_budget)
                if self.cfg.thinking_budget is not None
                else None
            ),
            response_schema=response_schema if use_json_mode else None,
            stop_sequences=self.cfg.stop_sequences,
            # response_mime_type="application/json"
        )

        start_time = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self._client().models.generate_content(
                    model=self.config.model,
                    contents=conversation_messages,
                    config=generate_content_config,
                )
                duration = time.time() - start_time
                metrics = self._extract_metrics(response, duration)
                text = response.text
                if text is None:
                    raise ValueError("Response text is None")
                self._end_call(call_id, text, extra={"metrics": metrics} if metrics else None)
                result: Dict[str, Any] = {"text": text}

                return result
            except Exception as e:
                last_err = e
                if attempt > self.config.max_retries:
                    self.logger.warning(f"Error at attempt {attempt}: Max retries reached, stopping retries")
                    break
                else:
                    self.logger.warning(f"Warning at attempt {attempt}: Retrying to call Gemini: {e}")
                delay = min(self.config.starting_delay * (self.config.backoff_factor ** attempt), self.config.max_delay)
                time.sleep(delay)
        # On failure, record error
        self._end_call(call_id, "", extra={"error": str(last_err)})
        return {"text": ""}

    def _extract_metrics(self, response: Any, duration: float) -> Optional[Dict[str, Any]]:
        try:
            usage = getattr(response, "usage_metadata", None)
            return {
                "duration": duration,
                "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                "thinking_tokens": getattr(usage, "thoughts_token_count", None) if usage else None,
                "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
            }
        except Exception:
            return None

    def _client(self) -> genai.Client:
        if self._gemini_client is None:
            scopes = [
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/cloud-platform",
            ]

            credentials_path = os.getenv("GCP_CREDENTIALS")
            project_id = os.getenv("GCP_PROJECT")
            region = os.getenv("GCP_REGION")

            if not credentials_path:
                raise ValueError("GCP_CREDENTIALS environment variable not set")
            if not project_id:
                raise ValueError("GCP_PROJECT environment variable not set")
            if not region:
                raise ValueError("GCP_REGION environment variable not set")

            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )

            self._gemini_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
                credentials=credentials,
            )
        return self._gemini_client