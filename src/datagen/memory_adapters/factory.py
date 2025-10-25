from typing import Union, Dict, Any, Optional
from logging import Logger

from src.datagen.memory_adapters.memory_adapter import MemoryAdapter, MemoryAdapterConfig
from src.datagen.memory_adapters.history_list_adapter import HistoryAdapter, HistoryAdapterConfig


def build_memory_adapter(config: Union[MemoryAdapterConfig, Dict[str, Any]], logger: Optional[Logger] = None) -> MemoryAdapter:
    if isinstance(config, dict):
        type = config.get("type")
        cfg_dict = config.copy()
    else:
        type = config.type
        cfg_dict = config.model_dump(exclude_unset=True)
    
    if type == "history_list":
        if isinstance(config, HistoryAdapterConfig):
            return HistoryAdapter(config, logger=logger)
        else:
            return HistoryAdapter(HistoryAdapterConfig(**cfg_dict), logger=logger)
    raise ValueError(f"Memory adapter type {type} not supported.")