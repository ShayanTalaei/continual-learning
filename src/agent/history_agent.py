from typing import List
from pydantic import BaseModel

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry
from src.memory.memory_factory import build_memory


class HistoryAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig
    history_k: int | None = None


class HistoryAgent(MemoryAgent):
    def __init__(self, config: HistoryAgentConfig, logger=None):
        super().__init__(config, logger=logger)

    def build_system_prompt(self) -> str:
        return "You are a helpful QA assistant. Use prior history when useful."

    def build_user_prompt(self, obs: str, history: List[Entry], k: int | None) -> str:
        lines: List[str] = []
        recent = history[-k:] if k is not None else history
        for entry in recent:
            entry_type = entry.type.upper()
            lines.append(f"{entry_type}: {entry.content}")
        lines.append(f"Q: {obs}")
        return "\n\n".join(lines)

    def update_memory_with_entry(self, entry: Entry) -> None:
        self.memory.update(entry)

