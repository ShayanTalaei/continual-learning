from typing import Iterable, List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel

from src.memory.history_list import Entry


class SampleSpec(BaseModel):
    observation: str
    target_action: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None
    focus_key: Optional[Tuple[int, int]] = None  # (episode_id, step_id)
    memory_directives: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    sample_id: Optional[str] = None  # Unique identifier for this sample


class MemoryFormationStrategy:
    def iter_samples(self, full_history: List[Entry]) -> Iterable[SampleSpec]:
        raise NotImplementedError

    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        raise NotImplementedError


