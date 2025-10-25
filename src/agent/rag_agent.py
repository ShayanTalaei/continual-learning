from typing import List, Any, Union, Dict, Tuple

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import Entry
from src.memory.vector_db import VectorDBConfig


class RAGAgentConfig(MemoryAgentConfig):
    memory_config: VectorDBConfig  # type: ignore[assignment]
    top_k: int = 5
    include_actions: bool = True
    include_feedback: bool = True
    flipping_examples: bool = False
    # Filtering controls
    filter_exact_task: bool = False
    filter_normalizer: Union[str, None] = None  # e.g., "lower_ws_collapse", "strip_lower"


class RAGAgent(MemoryAgent):
    def __init__(self, config: RAGAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.config: RAGAgentConfig = config
        self._pending_observation: Union[str, None] = None
        self._pending_action: Union[str, None] = None
        self._expect_pre_action_obs: bool = True

    def build_system_prompt(self) -> str:
        base = self.system_prompt
        return base

    def build_user_prompt(self, obs: str, history: List[Any], k: Union[int, None]) -> List[Dict[str, str]]:
        messages: List[dict] = []

        # Retrieve similar past experiences and format as alternating chat turns
        retrieved: List[Tuple[Any, float]] = []
        query_fn = getattr(self.memory, "query", None)
        if query_fn is not None:
            try:
                retrieved = query_fn(obs, getattr(self.config, "top_k", 5))
            except Exception:
                retrieved = []

        # Optional filtering: exclude exact task matches
        if self.config.filter_exact_task and retrieved:
            def _normalize(text: str) -> str:
                t = text or ""
                norm = self.config.filter_normalizer or ""
                if norm == "lower_ws_collapse":
                    return " ".join(str(t).lower().split())
                if norm == "strip_lower":
                    return str(t).strip().lower()
                return str(t)

            obs_n = _normalize(obs)
            filtered: List[Tuple[Any, float]] = []
            for rec, score in retrieved:
                try:
                    key_text = getattr(rec, "key_text", None)
                    if key_text is None and hasattr(rec, "value"):
                        key_text = getattr(rec, "value", {}).get("observation")  # type: ignore[attr-defined]
                    if _normalize(str(key_text)) == obs_n:
                        continue
                except Exception:
                    pass
                filtered.append((rec, score))
            retrieved = filtered

        if self.config.flipping_examples:
            retrieved = list(reversed(retrieved))

        for rec, _ in retrieved:
            val = getattr(rec, "value", {})
            prev_obs = str(val.get("observation", ""))
            if prev_obs:
                messages.append({"role": "user", "content": prev_obs})
            if self.config.include_actions:
                act = val.get("action")
                if act is not None:
                    messages.append({"role": "assistant", "content": str(act)})
            if self.config.include_feedback:
                fb = val.get("feedback")
                if fb is not None:
                    messages.append({"role": "user", "content": f"Feedback: {str(fb)}"})

        # Current observation as final user message
        messages.append({"role": "user", "content": f"{obs}"})
        return messages

    # Grouped ingestion: buffer obs+action and write once at feedback time
    def create_observation_event(self, obs: str) -> Any:
        if self._expect_pre_action_obs:
            self._pending_observation = obs
            self._expect_pre_action_obs = False
        return None

    def create_action_event(self, action: str) -> Any:
        self._pending_action = action
        return None

    def create_feedback_event(self, feedback: dict) -> Any:
        fb = feedback.get("message", "") if isinstance(feedback, dict) else str(feedback)
        experience = {
            "observation": self._pending_observation or "",
            "action": self._pending_action,
            "feedback": fb,
        }
        # Reset buffer
        self._pending_observation = None
        self._pending_action = None
        self._expect_pre_action_obs = True
        return experience


