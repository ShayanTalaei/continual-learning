from typing import List, Any, Union, Dict

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry


class HistoryAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: Union[int, None] = None


class HistoryAgent(MemoryAgent):
    def __init__(self, config: HistoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)

    def build_system_prompt(self) -> str:
        history_list_instructions = ("You will be given the previous experiences you've had and their feedback. "
            "You should use this feedback to improve your performance in the subsequent actions.")
        
        return self.system_prompt + "\n\n" + history_list_instructions

    def build_user_prompt(self, obs: str, history: List[Any], k: Union[int, None]) -> List[Dict[str, str]]:
        messages: List[dict] = []
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        
        messages.append({"role": "user", "content": "Here are the previous experiences you've had and their feedback:"})
        # Add previous experiences as alternating user/assistant messages
        for entry in recent:
            if entry.type == "Observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type == "Action":
                messages.append({"role": "assistant", "content": str(entry.content)})
            elif entry.type == "Feedback":
                # Add feedback as a user message
                messages.append({"role": "user", "content": f"Feedback: {entry.content}"})
        if len(recent) == 0:
            messages.append({"role": "user", "content": "No previous experiences."})
        
        # Add current observation as the final user message
        messages.append({"role": "user", "content": f"Here is the current observation: {obs}"})
        
        return messages

    def create_observation_event(self, obs: str) -> Any:
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        return Entry(type="Feedback", content=feedback.get("message", ""))

