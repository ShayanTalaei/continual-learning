from typing import List, Tuple, Dict, Any, Union, Optional
from pathlib import Path
import json
import threading

from pydantic import BaseModel

from src.memory.memory_module import MemoryModule, MemoryModuleConfig
from src.lm.embedding_model import EmbeddingConfig, EmbeddingModel
from src.lm.lm_factory import get_embedding_client


class VectorDBConfig(MemoryModuleConfig):
    _type: str = "vector_db"
    embedding_config: Union[EmbeddingConfig, Dict[str, Any]]
    distance: str = "cosine"  # or "dot"
    normalize: bool = True
    max_items: Optional[int] = None


class Record(BaseModel):
    id: int
    key_text: str
    value: Dict[str, Any]
    vector: List[float]
    meta: Optional[Dict[str, Any]] = None


class VectorDBMemory(MemoryModule):
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.config = config
        self._records: List[Record] = []
        self._matrix: List[List[float]] = []
        self._dim: Optional[int] = None
        self._next_id: int = 1
        self._lock = threading.Lock()
        self._embed: EmbeddingModel = get_embedding_client(config.embedding_config)

    def _update(self, experience: Dict[str, Any]):
        # Expect a grouped dict with observation (required), action/feedback optional
        key = str(experience.get("observation", ""))
        vec = self._embed.embed_one(key)
        if self.config.normalize:
            vec = self._l2_normalize(vec)
        with self._lock:
            if self._dim is None:
                self._dim = len(vec)
            rec = Record(
                id=self._next_id,
                key_text=key,
                value={k: v for k, v in experience.items() if k in ("observation", "action", "feedback", "meta")},
                vector=list(vec),
            )
            self._records.append(rec)
            self._matrix.append(list(vec))
            self._next_id += 1
            if self.config.max_items is not None and len(self._records) > self.config.max_items:
                # FIFO eviction
                self._records.pop(0)
                self._matrix.pop(0)

    def recall(self) -> List[Dict[str, Any]]:
        # Return shallow summaries
        with self._lock:
            return [{"id": r.id, "observation": r.value.get("observation"), "action": r.value.get("action"), "feedback": r.value.get("feedback")} for r in self._records]

    def query(self, query_text: str, top_k: int) -> List[Tuple[Record, float]]:
        q = self._embed.embed_one(query_text)
        if self.config.normalize:
            q = self._l2_normalize(q)
        with self._lock:
            if not self._records:
                return []
            scores = [self._similarity(q, v) for v in self._matrix]
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(top_k, 0)]
            return [(self._records[i], scores[i]) for i in idxs]

    def save_snapshot(self, base_dir: Union[str, Path], snapshot_id: Union[int, str]) -> str:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"vector_db_{snapshot_id}.jsonl"
        with open(path, "w") as f:
            for r in self._records:
                f.write(json.dumps(r.model_dump()) + "\n")
        return str(path)

    @classmethod
    def load_snapshot(cls, snapshot_path: Union[str, Path]) -> "VectorDBMemory":
        path = Path(snapshot_path)
        cfg = VectorDBConfig(embedding_config={"model": "text-embedding-004"})
        mem = cls(cfg)
        if not path.exists():
            return mem
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = Record(**json.loads(line))
                mem._records.append(rec)
                mem._matrix.append(list(rec.vector))
                mem._next_id = max(mem._next_id, rec.id + 1)
                if mem._dim is None:
                    mem._dim = len(rec.vector)
        return mem

    def _l2_normalize(self, vec: List[float]) -> List[float]:
        s = sum(v * v for v in vec) ** 0.5
        if s == 0:
            return list(vec)
        return [v / s for v in vec]

    def _similarity(self, a: List[float], b: List[float]) -> float:
        if self.config.distance == "dot":
            return sum(x * y for x, y in zip(a, b))
        # cosine: vectors are normalized already if normalize=True
        dot = sum(x * y for x, y in zip(a, b))
        if self.config.normalize:
            return dot
        # fall back cosine with norms
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


