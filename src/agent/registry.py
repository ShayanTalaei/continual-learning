from typing import Dict, Tuple, Type, Any

from src.agent.agent import AgentConfig, Agent
from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.agent.history_agent import HistoryAgent, HistoryAgentConfig
from src.agent.memoryless_agent import MemorylessAgent, MemorylessAgentConfig
from src.agent.reflexion_agent import ReflexionAgent, ReflexionAgentConfig
from src.agent.kv_memory_agent import KVMemoryAgent, KVMemoryAgentConfig
from src.agent.hindsight_agent import HindsightAgent, HindsightAgentConfig
from src.agent.appworld_generalist_agent import (
    AppWorldCUGA, AppWorldCUGAConfig,
    AppWorldPlanner, AppWorldPlannerConfig,
    AppWorldAPISubAgent, AppWorldAPISubAgentConfig,
    AppWorldAPICodePlanner, AppWorldAPICodePlannerConfig,
    AppWorldCodeAgent, AppWorldCodeAgentConfig,
)


AGENT_REGISTRY: Dict[str, Tuple[Type[AgentConfig], Type[Agent]]] = {
    "memory_agent": (MemoryAgentConfig, MemoryAgent),
    "history_agent": (HistoryAgentConfig, HistoryAgent),
    "memoryless_agent": (MemorylessAgentConfig, MemorylessAgent),
    "reflexion_agent": (ReflexionAgentConfig, ReflexionAgent),
    "kv_memory_agent": (KVMemoryAgentConfig, KVMemoryAgent),
    "hindsight_agent": (HindsightAgentConfig, HindsightAgent),
    "appworld_planner": (AppWorldPlannerConfig, AppWorldPlanner),
    "appworld_api_sub_agent": (AppWorldAPISubAgentConfig, AppWorldAPISubAgent),
    "appworld_code_planner": (AppWorldAPICodePlannerConfig, AppWorldAPICodePlanner),
    "appworld_code_agent": (AppWorldCodeAgentConfig, AppWorldCodeAgent),
    "appworld_cuga": (AppWorldCUGAConfig, AppWorldCUGA),
}


