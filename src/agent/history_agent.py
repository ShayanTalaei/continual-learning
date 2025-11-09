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
        collapse_messages: Union[bool, None] = None,
    ) -> List[Dict[str, str]]:
        config = cast(HistoryAgentConfig, self.config)
        collapse = config.collapse_messages if collapse_messages is None else collapse_messages
        recent_entries = cast(List[Entry], history[-k:] if k is not None else history)

        experiences = self._collect_experiences(recent_entries)
        if collapse:
            content = self._format_collapsed_prompt(experiences, obs)
            return [{"role": "user", "content": content}]

        messages = self._format_expanded_prompt(experiences)
        messages.append({"role": "user", "content": str(obs)})
        return messages

    def _collect_experiences(self, entries: List[Entry]) -> List[Dict[str, str]]:
        experiences: List[Dict[str, str]] = []
        current: Dict[str, str] = {}

        for entry in entries:
            etype = entry.type.lower()
            content = str(entry.content)

            if etype == "observation":
                if current:
                    experiences.append(current)
                    current = {}
                current["observation"] = content
            elif etype == "action":
                current["action"] = content
            elif etype == "feedback":
                current["feedback"] = content

        if current:
            experiences.append(current)

        return experiences

    def _format_collapsed_prompt(self, experiences: List[Dict[str, str]], obs: str) -> str:
        blocks = []
        for exp in experiences:
            lines = []
            observation = exp.get("observation")
            action = exp.get("action")
            feedback = exp.get("feedback")

            if observation:
                lines.append(f"Observation: {observation}")
            if action:
                lines.append(f"Action: {action}")
            if feedback:
                lines.append(f"Feedback: {feedback}")

            if lines:
                blocks.append("\n".join(lines))

        prior = "\n\n".join(blocks)
        parts = [prior] if prior else []
        parts.append(str(obs))
        return "\n\n".join(parts)

    def _format_expanded_prompt(self, experiences: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        for exp in experiences:
            observation = exp.get("observation")
            action = exp.get("action")
            feedback = exp.get("feedback")

            if observation:
                messages.append({"role": "user", "content": observation})
            if action:
                messages.append({"role": "assistant", "content": action})
            if feedback:
                messages.append({"role": "user", "content": feedback})

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


