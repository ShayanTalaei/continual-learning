from typing import List, Any
from abc import ABC, abstractmethod
from src.agent.agent import Agent, AgentConfig
from src.memory.memory_module import MemoryModule, MemoryModuleConfig
from src.memory.memory_factory import build_memory
from contextlib import contextmanager


class MemoryAgentConfig(AgentConfig):
    memory_config: MemoryModuleConfig
    history_k: int | None = None
    system_prompt: str | None = None
    verbose: bool = True

class MemoryAgent(Agent[MemoryAgentConfig], ABC):
    def __init__(self, config: MemoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        if config.lm_config is None:
            raise ValueError("MemoryAgent requires lm_config in config")
        self.memory: MemoryModule = build_memory(config.memory_config)
        self._last_action: str | None = None
        self._trajectory: List[Any] = []
        self.logger.info("MemoryAgent init: history_k=%s", self.config.history_k)

    def build_system_prompt(self) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        return "You are a helpful assistant. Use prior context when useful."

    @abstractmethod
    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> str:
        pass

    @abstractmethod
    def create_observation_event(self, obs: str) -> Any:
        pass

    @abstractmethod
    def create_action_event(self, action: str) -> Any:
        pass

    @abstractmethod
    def create_feedback_event(self, feedback: dict) -> Any:
        pass
    
    @contextmanager
    def eval_mode(self):
        prev = self.training
        memory_prev = self.memory.training
        try:
            self.training = False
            self.memory.training = False
            yield self
        finally:
            self.training = prev
            self.memory.training = memory_prev
            
    def act(self, obs: str) -> str:
        self.logger.info("Act: obs_len=%d", len(obs))
        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(obs, history, self.config.history_k)
        history_len = len(history[-self.config.history_k:]) if self.config.history_k is not None else len(history)
        self.logger.info("Prompt built: history_items=%d", history_len)

        obs_event = self.create_observation_event(obs)
        if obs_event is not None:
            self.memory.update(obs_event)
            self._trajectory.append(obs_event)
        self.logger.info("Logged Observation")
        
        action = self.lm.call(system_prompt, user_prompt)
        if action is None:
            self.logger.warning("No action returned from LM")
            action = ""
        action = action.strip()
        self.logger.info(f"Action generated: {action[:25] + '...' + action[-25:] if len(action) > 50 else action}")
        
        self._last_action = action
        action_event = self.create_action_event(self._last_action)
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)
        self.logger.info("Logged Action")
        
        return action

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        if self._last_action is None:
            return
        # Create a generic experience content string
        if obs is not None:
            obs_event = self.create_observation_event(obs)
            if obs_event is not None:
                self.memory.update(obs_event)
                self._trajectory.append(obs_event)
            self.logger.info("Logged Observation (post-step)")
        
        feedback_event = self.create_feedback_event(feedback)
        if feedback_event is not None:
            self.memory.update(feedback_event)
            # Track minimal trajectory as raw content
            self._trajectory.append(feedback_event)
        self.logger.info("Logged Feedback correct=%s", str(feedback.get("correct")))

    def end_episode(self) -> None:
        # No reflections in the minimal MemoryAgent
        self.logger.info("End episode: trajectory_len=%d", len(self._trajectory))
        self._trajectory = []

    def clone_for_episode(self, training: bool, share_memory: bool = True) -> "MemoryAgent":
        clone = self.__class__(self.config, logger=self.logger)
        # Share LM to save resources
        clone.lm = self.lm
        # Optionally share memory (safe for eval when training=False)
        if share_memory:
            clone.memory = self.memory
        clone.training = training
        self.memory.training = training
        clone._trajectory = []
        clone._last_action = None
        return clone