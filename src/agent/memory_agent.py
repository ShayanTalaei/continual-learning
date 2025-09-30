from typing import List
from src.agent.agent import Agent, AgentConfig
from src.memory.memory_module import MemoryModule, MemoryModuleConfig
from src.memory.history_list import Entry
from src.memory.memory_factory import build_memory
from src.lm.language_model import LMConfig, LanguageModel
from src.lm.lm_factory import get_lm_client


class MemoryAgentConfig(AgentConfig):
    lm_config: LMConfig
    memory_config: MemoryModuleConfig
    history_k: int | None = None
    system_prompt: str | None = None
    verbose: bool = True

class MemoryAgent(Agent[MemoryAgentConfig]):
    def __init__(self, config: MemoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.memory: MemoryModule = build_memory(config.memory_config)
        self.lm: LanguageModel = get_lm_client(config.lm_config)
        self._last_action: str | None = None
        self._trajectory: List[Entry] = []
        self.logger.info("MemoryAgent init: history_k=%s", self.config.history_k)

    def build_system_prompt(self) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        return "You are a helpful assistant. Use prior context when useful."

    def build_user_prompt(self, obs: str, history: List[Entry], k: int | None) -> str:
        lines: List[str] = []
        # format last k entries
        recent = history[-k:] if k is not None else history
        for entry in recent:
            entry_type = entry.type.upper() if hasattr(entry, "type") else "ENTRY"
            lines.append(f"{entry_type}: {entry.content}")
        lines.append(f"Q: {obs}")
        return "\n\n".join(lines)

    def act(self, obs: str) -> str:
        self.logger.info("Act: obs_len=%d", len(obs))
        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(obs, history, self.config.history_k)
        history_len = len(history[-self.config.history_k:]) if self.config.history_k is not None else len(history)
        self.logger.info("Prompt built: history_items=%d", history_len)

        entry = Entry(type="Observation", content=obs)
        self.update_memory_with_entry(entry)
        self._trajectory.append(entry)
        self.logger.info("Logged Observation")
        
        action = self.lm.call(system_prompt, user_prompt)
        if action is None:
            self.logger.warning("No action returned from LM")
            action = ""
        action = action.strip()
        self.logger.info(f"Action generated: {action[:25] + '...' + action[-25:] if len(action) > 50 else action}")
        
        self._last_action = action
        action_entry = Entry(type="Action", content=self._last_action)
        self.update_memory_with_entry(action_entry)
        self._trajectory.append(action_entry)
        self.logger.info("Logged Action")
        
        return action

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        if self._last_action is None:
            return
        # Create a generic experience content string
        if obs is not None:
            obs_entry = Entry(type="Observation", content=obs)
            self.update_memory_with_entry(obs_entry)
            self._trajectory.append(obs_entry)
            self.logger.info("Logged Observation (post-step)")
        
        feedback_entry = Entry(type="Feedback", content=feedback.get('message', ''))
        self.update_memory_with_entry(feedback_entry)
        # Track minimal trajectory as raw content
        self._trajectory.append(feedback_entry)
        self.logger.info("Logged Feedback correct=%s", str(feedback.get("correct")))

    def update_memory_with_entry(self, entry: Entry) -> None:
        self.memory.update(entry)

    def end_episode(self) -> None:
        # No reflections in the minimal MemoryAgent
        self.logger.info("End episode: trajectory_len=%d", len(self._trajectory))
        self._trajectory = []