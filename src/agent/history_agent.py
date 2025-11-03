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
        return self.system_prompt

    def build_user_prompt(self, obs: str, history: List[Any], k: Union[int, None], merge_feedback_and_observation: bool = False) -> List[Dict[str, str]]:
        messages: List[dict] = []
        recent: List[Entry] = history[-k:] if k is not None else history  # type: ignore[assignment]
        
        # Combine all prior experiences into a single user message, each experience as a block (Observation, Action, Feedback)
        experiences = []
        current_experience = {}
        # Gather sequentially as episodes: observation -> action -> feedback
        for entry in recent:
            etype = entry.type.lower()
            if etype == "observation":
                # Start a new experience
                if current_experience:
                    experiences.append(current_experience)
                    current_experience = {}
                current_experience["observation"] = str(entry.content)
            elif etype == "action":
                content_str = str(entry.content)
                # if "</think>" in content_str:
                #     content_str = content_str.split("</think>", 1)[1]
                current_experience["action"] = content_str
            elif etype == "feedback":
                current_experience["feedback"] = f"{entry.content}"
        if current_experience:
            experiences.append(current_experience)

        exp_blocks = []
        for i, exp in enumerate(experiences):
            lines = []
            if "observation" in exp:
                lines.append(f"Observation: {exp['observation']}")
            if "action" in exp:
                lines.append(f"Action: {exp['action']}")
            if "feedback" in exp:
                lines.append(f"Feedback: {exp['feedback']}")
            exp_blocks.append("\n".join(lines))
        user_content = "Here are the previous experiences you've had and their feedback:\n\n"
        user_content += "\n\n".join(exp_blocks)
        user_content += f"\n\nHere is the current observation: {obs}"
        messages.append({"role": "user", "content": user_content})

        
        return messages

    def create_observation_event(self, obs: str) -> Any:
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        # If the action contains \boxed{...}, extract whatever is inside the boxed
        start = action.find('\\boxed{')
        if start != -1:
            start += len('\\boxed{')
            end = action.find('}', start)
            if end != -1:
                action = f"\\boxed{{{action[start:end]}}}"
            else:
                action = ""
        else:
            action = ""
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        return Entry(type="Feedback", content=feedback.get("message", ""))


