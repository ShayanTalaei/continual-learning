from typing import List, Literal, Union, Dict, Any
from pathlib import Path
import json
from pydantic import BaseModel
from src.memory.memory_module import MemoryModule, MemoryModuleConfig

class Entry(BaseModel):
    type: str
    content: str|dict


class HistoryListConfig(MemoryModuleConfig):
    _type: Literal["history_list"] = "history_list"
    max_length: int | None = None

class HistoryList(MemoryModule):
    def __init__(self, config: HistoryListConfig):
        super().__init__(config)
        self.history_list: List[Entry] = []

    def _update(self, entry: Entry):
        self.history_list.append(entry)
        if self.config.max_length is not None and len(self.history_list) > self.config.max_length:
            self.history_list.pop(0)

    def recall(self):
        return self.history_list

    # Snapshot implementations
    def save_snapshot(self, base_dir: Union[str, Path], snapshot_id: Union[int, str]) -> str:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        # memory_{id}.jsonl
        path = base / f"memory_{snapshot_id}.jsonl"
        with open(path, "w") as f:
            for entry in self.history_list:
                f.write(json.dumps(entry.model_dump()) + "\n")
        return str(path)

    @classmethod
    def load_snapshot(cls, snapshot_path: Union[str, Path]) -> "HistoryList":
        path = Path(snapshot_path)
        config = HistoryListConfig()
        mem = cls(config)
        if not path.exists():
            return mem
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        mem.history_list.append(Entry(**rec))
                    except Exception:
                        continue
        except Exception:
            pass
        return mem