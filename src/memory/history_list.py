from typing import List, Literal, Union, Dict, Any
from pydantic import BaseModel
from src.memory.memory_module import MemoryModule, MemoryModuleConfig

class Entry(BaseModel):
    type: str
    content: str


class HistoryListConfig(MemoryModuleConfig):
    _type: Literal["history_list"] = "history_list"
    max_length: int | None = None

class HistoryList(MemoryModule):
    def __init__(self, config: HistoryListConfig):
        super().__init__(config)
        self.history_list: List[Entry] = []

    def update(self, entry: Entry):
        self.history_list.append(entry)
        if self.config.max_length is not None and len(self.history_list) > self.config.max_length:
            self.history_list.pop(0)

    def recall(self):
        return self.history_list