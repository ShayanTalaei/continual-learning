from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
from threading import Lock
from datetime import datetime
from src.utils import logger as jsonlogger
from logging import Logger, getLogger

class LMConfig(BaseModel):
    model: str
    train_temperature: float = 0.2
    val_temperature: float = 0.2
    max_output_tokens: int = 8192
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
        self._lock: Lock = Lock()
        # Track call_id to path via the generic logger return
        self._call_paths: Dict[str, Path] = {}
        self.logger = logger or getLogger("language_model")
        
    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call the language model with a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Common roles: 'system', 'user', 'assistant'
        
        Returns:
            Dictionary containing 'text' and optionally 'metrics' and 'logprobs'
        """
        raise NotImplementedError

    def _begin_call(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self.config.log_calls:
            return None
        ctx: Dict[str, Any] = jsonlogger.json_get_context()
        # Build subdirs based on context: mode/call_type structure
        subdirs = []
        mode = ctx.get("mode")
        call_type = ctx.get("call_type")  # e.g., "reflection"
        
        # Always start with mode
        if mode == "val" or mode == "validation":
            # Nest validation under validation/val_{num_seen_episodes}
            num_seen_episodes = ctx.get("num_seen_episodes")
            subdirs.append("validation")
            if num_seen_episodes is not None:
                subdirs.append(f"val_{int(num_seen_episodes)}")
        elif mode:
            subdirs.append(str(mode))
        else:
            # Default mode if none specified
            subdirs.append("default")
        
        # Add call_type subdirectory under mode (e.g., train/reflections/)
        if call_type:
            subdirs.append(f"{call_type}s")
        else:
            # Default call_type for regular action generation
            subdirs.append("actions")
        
        # Filename adds episode and step if available
        episode_index = ctx.get("episode_index")
        step_index = ctx.get("step_index")
        filename = None
        payload = {
            "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"),
            "model": getattr(self.config, "model", None),
            "messages": messages,
            "context": ctx,
        }
        call_id = jsonlogger.json_next_id()
        # Compose filename with call_type, episode and step if present
        parts = []
        if call_type:
            parts.append(call_type)
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
        if not call_id or not self.config.log_calls:
            return
        with self._lock:
            path = self._call_paths.get(call_id, (jsonlogger.json_ensure_dir() / f"{call_id}.json"))
        updates: Dict[str, Any] = {
            "output": output,
            "timestamp_end": datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"),
        }
        if extra:
            updates["extra"] = extra
        jsonlogger.json_record_update(path, updates)