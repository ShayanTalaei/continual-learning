from typing import List

from src.datagen.strategies.strategy import StrategyConfig
from src.datagen.strategies.strategy import Strategy
from src.datagen.memory_adapters.factory import build_memory_adapter
from src.datagen.types import GenerationItem
from src.datagen.memory_adapters.memory_adapter import MemoryAdapterConfig

class ReplayStrategyConfig(StrategyConfig):
    memory_adapter: MemoryAdapterConfig

class ReplayStrategy(Strategy):
    def __init__(self, config: ReplayStrategyConfig, logger=None):
        super().__init__(config, logger)
        self.memory_adapter = build_memory_adapter(config.memory_adapter)

    def generate(self) -> List[GenerationItem]:
        return []