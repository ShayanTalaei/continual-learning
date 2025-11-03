from typing import List, Any, Union, Dict, cast

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry


class HistoryAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: Union[int, None] = None
    collapse_messages: bool = True
    only_boxed: bool = True


class HistoryAgent(MemoryAgent):
    def __init__(self, config: HistoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.config: HistoryAgentConfig = config

    def build_system_prompt(self) -> str:
        return self.system_prompt

    def build_user_prompt(
        self,
        obs: str,
        history: List[Any],
        k: Union[int, None],
        merge_feedback_and_observation: bool = False,
        collapse_messages: Union[bool, None] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        config = cast(HistoryAgentConfig, self.config)
        collapse = config.collapse_messages if collapse_messages is None else collapse_messages
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
        if collapse:
            user_content = "Here are the previous experiences you've had and their feedback:\n\n"
            user_content += "\n\n".join(exp_blocks)
            user_content += f"\n\nHere is the current observation: {obs}"
            messages.append({"role": "user", "content": user_content})
        else:
            if experiences:
                messages.append({"role": "user", "content": "Here are the previous experiences you've had and their feedback."})
                for exp in experiences:
                    obs_text = exp.get("observation")
                    action_text = exp.get("action")
                    feedback_text = exp.get("feedback")

                    if obs_text:
                        if merge_feedback_and_observation and feedback_text:
                            messages.append({
                                "role": "user",
                                "content": f"Observation: {obs_text}\nFeedback: {feedback_text}",
                            })
                            feedback_text = None
                        else:
                            messages.append({"role": "user", "content": f"Observation: {obs_text}"})

                    if action_text:
                        messages.append({"role": "assistant", "content": f"{action_text}"})

                    if feedback_text:
                        messages.append({"role": "user", "content": f"Feedback: {feedback_text}"})

            messages.append({"role": "user", "content": f"Here is the current observation: {obs}"})

        return messages

    def create_observation_event(self, obs: str) -> Any:
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        if self.config.only_boxed:
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


