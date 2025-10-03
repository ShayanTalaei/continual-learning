from typing import Optional, List
from src.agent.agent import Agent, AgentConfig


class MemorylessAgentConfig(AgentConfig):
    pass


class MemorylessAgent(Agent[MemorylessAgentConfig]):
    def __init__(self, config: MemorylessAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        if config.lm_config is None:
            raise ValueError("MemorylessAgent requires lm_config in AgentConfig")
        self._trajectory: List[str] = []

    def build_system_prompt(self) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        return "You are a helpful assistant."

    def build_user_prompt(self, obs: str) -> str:
        return f"Q: {obs}"

    def act(self, obs: str) -> str:
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(obs)
        action = self.lm.call(system_prompt, user_prompt) or ""
        action = action.strip()
        self._trajectory = [obs, action]
        return action

    def observe(self, obs: Optional[str], feedback: dict, done: bool) -> None:
        # No persistent memory; just clear ephemeral trajectory after using
        self._trajectory = []

    def reset(self) -> None:
        self._trajectory = []

    def end_episode(self) -> None:
        self._trajectory = []

    def clone_for_episode(self, training: bool, share_memory: bool = True) -> "MemorylessAgent":
        clone = MemorylessAgent(self.config, logger=self.logger)
        clone.lm = self.lm
        clone.training = training
        clone._trajectory = []
        return clone


