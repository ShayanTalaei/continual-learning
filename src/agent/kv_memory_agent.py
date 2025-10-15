from typing import List, Any

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.kv_cache import KVCacheMemory, KVCacheMemoryConfig


class KVMemoryAgentConfig(MemoryAgentConfig):
    memory_config: KVCacheMemoryConfig
    history_k: int | None = None  # unused: kept for interface compatibility
    system_prompt: str | None = None
    verbose: bool = True


class KVMemoryAgent(MemoryAgent):
    def __init__(self, config: KVMemoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)

    def build_system_prompt(self) -> str:
        if self.system_prompt is not None:
            return self.system_prompt
        return "You are a helpful assistant. Use the attached cartridge as your long-term memory."

    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> str:
        # KV-only: keep prompt minimal to rely on the cartridge for context
        return f"Observation:\n{obs}"

    def create_observation_event(self, obs: str) -> Any:
        return None

    def create_action_event(self, action: str) -> Any:
        return None

    def create_feedback_event(self, feedback: dict) -> Any:
        return None

    def act(self, obs: str) -> str:
        self.logger.info("KV act: obs_len=%d", len(obs))
        recall = self.memory.recall()
        cartridges = recall.get("cartridges")
        if not cartridges:
            raise ValueError("KVMemoryAgent requires memory.recall() to provide a non-empty 'cartridges' list.")

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(obs, [], None)
        self.logger.info("KV prompt built; delegating to LM with cartridges=%s", str(cartridges))

        resp = self.lm.call(system_prompt, user_prompt, cartridges=cartridges)  # type: ignore[arg-type]
        action = (resp.get("text") or "").strip()
        if not action:
            self.logger.warning("No action returned from LM")
        self.logger.info("KV action generated: %s", action[:100])
        self._last_action = action
        return action


