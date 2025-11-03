from typing import Dict, Tuple, Type, Any

from src.agent.agent import AgentConfig, Agent
from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.agent.history_agent import HistoryAgent, HistoryAgentConfig
from src.agent.memoryless_agent import MemorylessAgent, MemorylessAgentConfig
from src.agent.reflexion_agent import ReflexionAgent, ReflexionAgentConfig
from src.agent.hindsight_agent import HindsightAgent, HindsightAgentConfig
from src.agent.rag_agent import RAGAgent, RAGAgentConfig
from src.agent.ace_agent import AceAgent, AceAgentConfig
from src.agent.system_summary_agent import SystemSummaryAgent, SystemSummaryAgentConfig


AGENT_REGISTRY: Dict[str, Tuple[Type[AgentConfig], Type[Agent]]] = {
    "memory_agent": (MemoryAgentConfig, MemoryAgent),
    "history_agent": (HistoryAgentConfig, HistoryAgent),
    "memoryless_agent": (MemorylessAgentConfig, MemorylessAgent),
    "reflexion_agent": (ReflexionAgentConfig, ReflexionAgent),
    "hindsight_agent": (HindsightAgentConfig, HindsightAgent),
    "rag_agent": (RAGAgentConfig, RAGAgent),
    "ace_agent": (AceAgentConfig, AceAgent),
    "system_summary_agent": (SystemSummaryAgentConfig, SystemSummaryAgent),
}


