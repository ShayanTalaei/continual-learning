from typing import Any
from pydantic import BaseModel

class MemoryModuleConfig(BaseModel):
    _type: str


class MemoryModule:
    def __init__(self, config: MemoryModuleConfig):
        self.config = config

    def update(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def recall(self, *args: Any, **kwargs: Any):
        raise NotImplementedError