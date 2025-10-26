import os
import time
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerationConfig
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from logging import Logger
from src.utils import logger as jsonlogger

from .language_model import LMConfig, LanguageModel

# Import freezegun bypass utilities for AppWorld compatibility
try:
    # AppWorld provides a low-level bypass that uses ctypes to get real system time
    from appworld.common.time import freezegun_bypassed_datetime
    APPWORLD_BYPASS_AVAILABLE = True
except ImportError:
    APPWORLD_BYPASS_AVAILABLE = False
    
    # Fallback: try to import freezegun directly
    try:
        from freezegun.api import real_time, real_datetime, real_date
        FREEZEGUN_AVAILABLE = True
    except ImportError:
        FREEZEGUN_AVAILABLE = False

# Load .env from the same directory as this file
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=_env_path, override=True)


@contextmanager
def _bypass_freezegun():
    """
    Temporarily restore real system time for JWT token generation.
    
    This is critical for AppWorld environments which use freezegun to mock time.
    Google's OAuth2 servers reject JWT tokens with timestamps far from real time.
    
    Uses monkey-patching to replace time functions with real implementations
    that use ctypes to bypass freezegun.
    """
    if not (APPWORLD_BYPASS_AVAILABLE or FREEZEGUN_AVAILABLE):
        yield
        return
    
    # Monkey-patch time module functions with real implementations
    import time as time_module
    import datetime as dt_module
    
    # Save frozen versions
    saved_time = time_module.time
    saved_datetime_now = dt_module.datetime.now
    saved_datetime_utcnow = dt_module.datetime.utcnow
    
    try:
        if APPWORLD_BYPASS_AVAILABLE:
            # Use AppWorld's ctypes-based bypass
            def real_time_func():
                return freezegun_bypassed_datetime().timestamp()
            
            def real_now_func(tz=None):
                dt = freezegun_bypassed_datetime()
                if tz:
                    import pytz
                    return dt.replace(tzinfo=pytz.UTC).astimezone(tz)
                return dt
            
            def real_utcnow_func():
                return freezegun_bypassed_datetime()
            
            time_module.time = real_time_func
            dt_module.datetime.now = real_now_func
            dt_module.datetime.utcnow = real_utcnow_func
        
        yield
    finally:
        # Restore frozen versions
        time_module.time = saved_time
        dt_module.datetime.now = saved_datetime_now
        dt_module.datetime.utcnow = saved_datetime_utcnow


class GeminiConfig(LMConfig):
    thinking_budget: Optional[int] = None
    stop_sequences: Optional[List[str]] = ["FEEDBACK", "OBSERVATION"]

class GeminiClient(LanguageModel):
    """Synchronous Gemini client compatible with `LanguageModel` interface."""

    def __init__(self, config: GeminiConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        self._gemini_client: Optional[genai.Client] = None
        self._credentials: Optional[service_account.Credentials] = None

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
            temperature=self.config.train_temperature,
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
                # Refresh credentials before each API call to prevent expiration
                self._refresh_credentials()
                
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

    def _refresh_credentials(self) -> None:
        """Refresh OAuth2 access token to prevent expiration.
        
        Uses real system time (bypassing freezegun) to ensure JWT tokens
        have valid timestamps that Google's OAuth2 servers will accept.
        """
        if self._credentials is not None:
            try:
                # Bypass freezegun to use real system time for JWT generation
                with _bypass_freezegun():
                    self._credentials.refresh(Request())
                self.logger.debug("Gemini credentials refreshed successfully")
            except Exception as e:
                self.logger.warning(f"Failed to refresh Gemini credentials: {e}")
                raise

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

            # Expand path variables and ~
            credentials_path = os.path.expanduser(os.path.expandvars(credentials_path))
            
            # Store credentials for later refresh
            self._credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
            
            # Explicitly refresh credentials to mint a valid OAuth2 access token
            # This prevents "invalid_grant: Invalid JWT" errors due to timing issues
            # Use real system time (bypass freezegun) for JWT generation
            try:
                with _bypass_freezegun():
                    self._credentials.refresh(Request())
                self.logger.info("Gemini credentials refreshed successfully during initialization")
            except Exception as e:
                self.logger.warning(f"Gemini credential initial refresh failed: {e}")
                raise

            self._gemini_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
                credentials=self._credentials,
            )
        return self._gemini_client