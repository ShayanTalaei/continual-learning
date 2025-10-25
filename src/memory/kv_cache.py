from typing import Dict, Any, Union
from pathlib import Path
import json
from pydantic import BaseModel

from src.memory.memory_module import MemoryModule, MemoryModuleConfig


class KVArtifact(BaseModel):
    id: str
    source: str  # one of {"local", "huggingface", "wandb"}
    force_redownload: bool = False


class KVCacheMemoryConfig(MemoryModuleConfig):
    _type: str = "kv_cache"
    artifact: KVArtifact
    tokenizer_name: str | None = None
    model_name: str | None = None


class KVCacheMemory(MemoryModule):
    def __init__(self, config: KVCacheMemoryConfig):
        super().__init__(config)
        self.config = config

    def _update(self, *args: Any, **kwargs: Any):
        raise RuntimeError("KVCacheMemory is immutable during inference; updates are not supported.")

    def recall(self) -> Dict[str, Any]:
        return {
            "type": "kv",
            "cartridges": [self.config.artifact.model_dump()],
        }

    def save_snapshot(self, base_dir: Union[str, Path], snapshot_id: Union[int, str]) -> str:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"kv_memory_{snapshot_id}.json"
        payload: Dict[str, Any] = {
            "type": "kv_cache",
            "artifact": self.config.artifact.model_dump(),
            "tokenizer_name": self.config.tokenizer_name,
            "model_name": self.config.model_name,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(path)

    @classmethod
    def load_snapshot(cls, snapshot_path: Union[str, Path]) -> "KVCacheMemory":
        path = Path(snapshot_path)
        with open(path, "r") as f:
            data = json.load(f)
        cfg = KVCacheMemoryConfig(
            artifact=KVArtifact(**data["artifact"]),
            tokenizer_name=data.get("tokenizer_name"),
            model_name=data.get("model_name"),
        )
        return cls(cfg)


