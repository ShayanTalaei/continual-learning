from src.memory.memory_module import MemoryModule, MemoryModuleConfig
from src.memory.history_list import HistoryList, HistoryListConfig


def build_memory(memory_config: MemoryModuleConfig) -> MemoryModule:
    if memory_config._type == "history_list":
        assert isinstance(memory_config, HistoryListConfig)
        return HistoryList(memory_config)
    raise ValueError(f"Memory type {memory_config._type} not supported.")