from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
from threading import Lock
from datetime import datetime
from src.utils import logger as jsonlogger
from logging import Logger, getLogger

class LMConfig(BaseModel):
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 2048
    log_calls: bool = False
    # Retry/backoff
    max_retries: int = 5
    starting_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 10.0
    
class LLMResponseMetrics(BaseModel):
    duration: float
    input_tokens: int
    thinking_tokens: int
    output_tokens: int
    total_tokens: int


class LanguageModel:
    def __init__(self, config: LMConfig, logger: Optional[Logger] = None):
        self.config = config
        self._log_calls: bool = False
        self._calls_dir: Optional[Path] = None
        self._lock: Lock = Lock()
        # Track call_id to path via the generic logger return
        self._call_paths: Dict[str, Path] = {}
        self.logger = logger or getLogger("language_model")
        
    def call(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    # Logging helpers (thread-safe)
    def enable_call_logging(self, calls_dir: str) -> None:
        self._log_calls = True
        self._calls_dir = Path(calls_dir)
        self._calls_dir.mkdir(parents=True, exist_ok=True)
        jsonlogger.enable_json_logging(str(self._calls_dir))

    def _begin_call(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not (self._log_calls and self._calls_dir):
            return None
        ctx: Dict[str, Any] = jsonlogger.json_get_context()
        # Build subdirs based on context
        subdirs = []
        mode = ctx.get("mode")
        if mode == "val" or mode == "validation":
            # Nest validation under validation/val_{num_seen_episodes}
            num_seen_episodes = ctx.get("num_seen_episodes")
            subdirs.append("validation")
            if num_seen_episodes is not None:
                subdirs.append(f"val_{int(num_seen_episodes)}")
        elif mode:
            subdirs.append(str(mode))
        # Filename adds episode and step if available
        episode_index = ctx.get("episode_index")
        step_index = ctx.get("step_index")
        filename = None
        payload = {
            "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"),
            "model": getattr(self.config, "model", None),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": ctx,
        }
        call_id = jsonlogger.json_next_id()
        # Compose filename with episode and step if present
        parts = []
        if episode_index is not None:
            parts.append(f"episode_{int(episode_index)}")
        if step_index is not None:
            parts.append(f"step_{int(step_index)}")
        parts.append(call_id)
        filename = "_".join(parts) + ".json"
        call_id, path = jsonlogger.json_record_begin(subdirs, filename, payload, supplied_id=call_id)
        with self._lock:
            self._call_paths[call_id] = path
        return call_id

    def _end_call(self, call_id: Optional[str], output: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not call_id or not (self._log_calls and self._calls_dir):
            return
        with self._lock:
            path = self._call_paths.get(call_id, (self._calls_dir / f"{call_id}.json"))
        updates: Dict[str, Any] = {
            "output": output,
            "timestamp_end": datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"),
        }
        if extra:
            updates["extra"] = extra
        jsonlogger.json_record_update(path, updates)