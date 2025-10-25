from src.memory.memory_module import MemoryModule, MemoryModuleConfig
from src.memory.history_list import HistoryList, HistoryListConfig
from src.memory.kv_cache import KVCacheMemory, KVCacheMemoryConfig


def build_memory(memory_config: MemoryModuleConfig) -> MemoryModule:
    if memory_config._type == "history_list":
        assert isinstance(memory_config, HistoryListConfig)
        return HistoryList(memory_config)
    if memory_config._type == "kv_cache":
        assert isinstance(memory_config, KVCacheMemoryConfig)
        return KVCacheMemory(memory_config)
    raise ValueError(f"Memory type {memory_config._type} not supported.")