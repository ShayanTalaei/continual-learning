from pydantic import BaseModel


class MemoryAdapterConfig(BaseModel):
    type: str
    checkpoint_path: str
    
class MemoryAdapter:
    
    def __init__(self, config: MemoryAdapterConfig, logger=None):
        self.config = config
        self.logger = logger
    
    def load(self) -> None:
        pass

