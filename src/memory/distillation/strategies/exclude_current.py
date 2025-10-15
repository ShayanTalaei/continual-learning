from typing import Iterable, List

from src.memory.history_list import Entry
from .base import MemoryFormationStrategy, SampleSpec


class ExcludeCurrentStrategy(MemoryFormationStrategy):
    def iter_samples(self, full_history: List[Entry]) -> Iterable[SampleSpec]:
        # A minimal pass that yields samples when we find Observation→Action→Feedback triples
        obs, act, fb = None, None, None
        for idx, e in enumerate(full_history):
            if e.type.lower() == "observation":
                obs = e.content if isinstance(e.content, str) else str(e.content)
                act, fb = None, None
            elif e.type.lower() == "action":
                act = e.content if isinstance(e.content, str) else str(e.content)
            elif e.type.lower() == "feedback":
                fb = e.content if isinstance(e.content, dict) else {"message": e.content}
                if obs is not None:
                    yield SampleSpec(
                        observation=obs,
                        target_action=act,
                        feedback=fb,
                        focus_key=(0, idx),  # episode/step indexing can be improved later
                        memory_directives={"exclude_span": (idx - 2, idx)},
                        meta={"triple_end_index": idx},
                    )
                obs, act, fb = None, None, None

    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        span = spec.memory_directives.get("exclude_span")
        if span is None:
            return full_history
        start, end = span
        return [e for i, e in enumerate(full_history) if i < start or i > end]


