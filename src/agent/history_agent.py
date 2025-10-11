from typing import List, Any

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry


class HistoryAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig
    history_k: int | None = None


class HistoryAgent(MemoryAgent):
    def __init__(self, config: HistoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)

    def build_system_prompt(self) -> str:
        history_list_instructions = ("You will be given the previous experiences you've had and their feedback. "
            "You should use this feedback to improve your performance in the subsequent actions.")
        
        return self.system_prompt + "\n\n" + history_list_instructions

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> str:
        lines: List[str] = []
        lines.append("Here is a list of your previous experiences:")
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        for entry in recent:
            entry_type = entry.type.upper()
            lines.append(f"{entry_type}: {entry.content}")
        if len(recent) == 0:
            lines.append("No previous experiences.")
        lines.append("Here is the current observation:")
        lines.append(f"{obs}")
        return "\n\n".join(lines)

    def create_observation_event(self, obs: str) -> Any:
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        return Entry(type="Feedback", content=feedback.get("message", ""))

