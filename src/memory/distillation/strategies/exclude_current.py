from typing import Iterable, List, Dict, Any
import random

from src.memory.history_list import Entry
from .base import MemoryFormationStrategy, SampleSpec


class ExcludeCurrentStrategy(MemoryFormationStrategy):
    def __init__(self, do_shuffle: bool = False, num_shufflings: int = 1):
        self.do_shuffle = do_shuffle
        self.num_shufflings = num_shufflings
    
    def _get_memory_snapshot(self, triplets: List[Dict[str, Any]]) -> List[Entry]:
        memory_snapshot = []
        for tri in triplets:
            memory_snapshot.append(Entry(
                type="Observation",
                content=tri['Observation'],
            ))
            memory_snapshot.append(Entry(
                type="Action",
                content=tri['Action'],
            ))
            memory_snapshot.append(Entry(
                type="Feedback",
                content=tri['Feedback'],
            ))
        return memory_snapshot
    
    def iter_samples(self, full_history: List[Entry]) -> Iterable[SampleSpec]:
        # First, collect all complete triplets (observation→action→feedback)
        triplets = []
        obs, act, fb = None, None, None
        obs_idx, act_idx, fb_idx = None, None, None
        
        for idx, e in enumerate(full_history):
            if e.type.lower() == "observation":
                obs = e.content
                obs_idx = idx
                act, fb = None, None
                act_idx, fb_idx = None, None
            elif e.type.lower() == "action":
                act = e.content
                act_idx = idx
            elif e.type.lower() == "feedback":
                fb = e.content
                fb_idx = idx
                if obs is not None and act is not None:
                    # We have a complete triplet
                    triplets.append({
                        'Observation': obs,
                        'Action': act,
                        'Feedback': fb,
                        'obs_idx': obs_idx,
                        'act_idx': act_idx,
                        'fb_idx': fb_idx
                    })
                obs, act, fb = None, None, None
                obs_idx, act_idx, fb_idx = None, None, None
        
        if not triplets:
            return  # No complete triplets found
        
        # If shuffling is enabled, create multiple shuffled versions
        for triplet in triplets:
            rest_of_triplets = triplets.copy()
            rest_of_triplets.remove(triplet)
            
            if self.do_shuffle:
                for shuffle_idx in range(self.num_shufflings):
                    shuffled_rest_of_triplets = rest_of_triplets.copy()
                    random.shuffle(shuffled_rest_of_triplets)
                    
                    sample_id = f"triplet_{triplet['fb_idx']}_shuffle_{shuffle_idx}"
                    yield SampleSpec(
                        observation=triplet['Observation'],
                        target_action=triplet['Action'],
                        feedback=triplet['Feedback'],
                        focus_key=(shuffle_idx, triplet['fb_idx']),  # shuffle_idx, step_id
                        memory_directives={"memory_snapshot": self._get_memory_snapshot(shuffled_rest_of_triplets)},
                        meta={"triple_end_index": triplet['fb_idx'], "shuffle_index": shuffle_idx},
                        sample_id=sample_id,
                    )
            else:
                sample_id = f"triplet_{triplet['fb_idx']}_shuffle_0"
                yield SampleSpec(
                    observation=triplet['Observation'],
                    target_action=triplet['Action'],
                    feedback=triplet['Feedback'],
                    focus_key=(0, triplet['fb_idx']),  # episode/step indexing
                    memory_directives={"memory_snapshot": self._get_memory_snapshot(rest_of_triplets)},
                    meta={"triple_end_index": triplet['fb_idx']},
                    sample_id=sample_id,
                )

    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        return spec.memory_directives["memory_snapshot"]


