from typing import Iterable, List, Dict, Any
from pydantic import BaseModel

from src.memory.history_list import Entry


class SampleSpec(BaseModel):
    observation: str
    target_action: str | None = None
    feedback: Dict[str, Any] | None = None
    focus_key: tuple[int, int] | None = None  # (episode_id, step_id)
    memory_directives: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}


class MemoryFormationStrategy:
    def iter_samples(self, full_history: List[Entry]) -> Iterable[SampleSpec]:
        raise NotImplementedError

    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        raise NotImplementedError


