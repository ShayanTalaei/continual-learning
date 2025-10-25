from typing import List
from pydantic import BaseModel

from src.datagen.types import GenerationItem


class StrategyConfig(BaseModel):
    type: str

class Strategy:
    def __init__(self, config: StrategyConfig, logger=None):
        self.config = config
        self.logger = logger

    def generate(self) -> List[GenerationItem]:
        raise NotImplementedError