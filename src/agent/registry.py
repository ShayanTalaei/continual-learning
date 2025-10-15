from typing import Dict, Tuple, Type, Any

from src.agent.agent import AgentConfig, Agent
from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.agent.history_agent import HistoryAgent, HistoryAgentConfig
from src.agent.memoryless_agent import MemorylessAgent, MemorylessAgentConfig
from src.agent.reflexion_agent import ReflexionAgent, ReflexionAgentConfig
from src.agent.kv_memory_agent import KVMemoryAgent, KVMemoryAgentConfig


AGENT_REGISTRY: Dict[str, Tuple[Type[AgentConfig], Type[Agent]]] = {
    "memory_agent": (MemoryAgentConfig, MemoryAgent),
    "history_agent": (HistoryAgentConfig, HistoryAgent),
    "memoryless_agent": (MemorylessAgentConfig, MemorylessAgent),
    "reflexion_agent": (ReflexionAgentConfig, ReflexionAgent),
    "kv_memory_agent": (KVMemoryAgentConfig, KVMemoryAgent),
}


