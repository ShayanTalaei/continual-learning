from typing import List

from src.datagen.strategies.strategy import StrategyConfig
from src.datagen.strategies.strategy import Strategy
from src.datagen.types import GenerationItem
from src.datagen.memory_adapters.memory_adapter import MemoryAdapterConfig
from src.datagen.memory_adapters.factory import build_memory_adapter


class SyntheticGenStrategyConfig(StrategyConfig):
    synthetic_gen_prompt_path: str
    memory_adapter: MemoryAdapterConfig
    
    
class SyntheticGenStrategy(Strategy):
    def __init__(self, config: SyntheticGenStrategyConfig, logger=None):
        super().__init__(config, logger)
        self.synthetic_gen_prompt_path = config.synthetic_gen_prompt_path
        self.memory_adapter = build_memory_adapter(config.memory_adapter)
        
    def generate(self) -> List[GenerationItem]:
        breakpoint()