from typing import List

from src.datagen.strategies.strategy import StrategyConfig
from src.datagen.strategies.strategy import Strategy
from src.datagen.types import GenerationItem
from src.datagen.memory_adapters.memory_adapter import MemoryAdapterConfig
from src.datagen.memory_adapters.factory import build_memory_adapter


class ReflectionStrategyConfig(StrategyConfig):
    reflection_prompt_path: str
    memory_adapter: MemoryAdapterConfig
    
    
class ReflectionStrategy(Strategy):
    def __init__(self, config: ReflectionStrategyConfig, logger=None):
        super().__init__(config, logger)
        self.reflection_prompt_path = config.reflection_prompt_path
        self.memory_adapter = build_memory_adapter(config.memory_adapter)
        
    def generate(self) -> List[GenerationItem]:
        return []