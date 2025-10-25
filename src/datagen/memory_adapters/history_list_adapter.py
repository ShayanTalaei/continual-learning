from src.datagen.memory_adapters.memory_adapter import MemoryAdapter, MemoryAdapterConfig
from src.memory.history_list import HistoryList
from pathlib import Path
from typing import List, Dict, Any, Optional


class HistoryAdapterConfig(MemoryAdapterConfig):
    pass

class HistoryAdapter(MemoryAdapter):
    def __init__(self, config: HistoryAdapterConfig, logger=None):
        super().__init__(config, logger)
        self.checkpoint_path = Path(config.checkpoint_path)
        
        self.history_list = HistoryList.load_snapshot(str(self.checkpoint_path))

    def _to_triplets(self) -> List[Dict[str, Any]]:
        triplets: List[Dict[str, Any]] = []
        obs: Optional[Any] = None
        act: Optional[Any] = None
        fb: Optional[Any] = None
        
        for e in self.history_list.recall():
            t = e.type.lower()
            if t == "observation":
                obs = e.content
                act, fb = None, None
            elif t == "action":
                act = e.content
            elif t == "feedback":
                fb = e.content
                if obs is not None and act is not None:
                    triplets.append({"Observation": obs, "Action": act, "Feedback": fb})
                obs, act, fb = None, None, None
        
        return triplets