from typing import Dict, Tuple, Type, Any

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.agent.history_agent import HistoryAgent, HistoryAgentConfig


AGENT_REGISTRY: Dict[str, Tuple[Type[MemoryAgentConfig], Type[MemoryAgent]]] = {
    "memory_agent": (MemoryAgentConfig, MemoryAgent),
    "history_agent": (HistoryAgentConfig, HistoryAgent),
}


