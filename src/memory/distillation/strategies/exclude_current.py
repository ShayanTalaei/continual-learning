from typing import Iterable, List, Dict, Any, Optional
import random

from src.memory.history_list import Entry
from .base import MemoryFormationStrategy, SampleSpec
from src.data.dataset_factory import build_dataset


class ExcludeCurrentStrategy(MemoryFormationStrategy):
    def __init__(self, do_shuffle: bool = False, num_shufflings: int = 1, subset_size: Optional[int] = None, shuffle_triplets: bool = False, target_dataset_config: Optional[Dict[str, Any]] = None, max_target_samples: Optional[int] = None, start_idx: Optional[int] = None, end_idx: Optional[int] = None, logger=None):
        self.do_shuffle = do_shuffle
        self.num_shufflings = num_shufflings
        # Optional subsampling of the rest-of-history triplets
        self.subset_size = subset_size
        # Optional one-shot shuffle of selected triplets before forming snapshot
        self.shuffle_triplets = shuffle_triplets
        # Optional target dataset for evaluation
        self.target_dataset_config = target_dataset_config
        self.max_target_samples = max_target_samples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.logger = logger

        # Lazily set by _load_target_dataset if provided
        if self.target_dataset_config is not None:
            self._load_target_dataset()

    def _load_target_dataset(self) -> None:
        if self.logger:
            self.logger.info(f"Loading target dataset with config: {self.target_dataset_config}")
        self.target_dataset = build_dataset(self.target_dataset_config, logger=self.logger)
        self.target_environments = self.target_dataset.get_dataset()
        # Apply optional slicing before max_target_samples
        if self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else None
            self.target_environments = self.target_environments[start:end]
        # Limit the number of samples if specified (applied after slicing)
        if self.max_target_samples is not None:
            self.target_environments = self.target_environments[:self.max_target_samples]
        if self.logger:
            self.logger.info(f"Loaded {len(self.target_environments)} target environments for evaluation")
    
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
                        'fb_idx': fb_idx//3
                    })
                obs, act, fb = None, None, None
                obs_idx, act_idx, fb_idx = None, None, None
        
        if not triplets:
            return  # No complete triplets found
        
        # For each current triplet, construct memory from the rest (with optional subsampling and shuffling)
        for i, triplet in enumerate(triplets):
            rest_of_triplets = triplets.copy()
            rest_of_triplets.remove(triplet)

            # Optional subsampling of the rest
            if self.subset_size is not None and len(rest_of_triplets) > self.subset_size:
                sampled_rest = random.sample(rest_of_triplets, self.subset_size)
            else:
                sampled_rest = rest_of_triplets

            if self.do_shuffle:
                for shuffle_idx in range(self.num_shufflings):
                    selected = sampled_rest.copy()
                    # Shuffle the selected set for each shuffle replication
                    random.shuffle(selected)
                    # Additionally allow explicit one-shot shuffle flag (redundant here but kept for parity)
                    if self.shuffle_triplets:
                        random.shuffle(selected)
                    sample_id = f"triplet_{triplet['fb_idx']}_shuffle_{shuffle_idx}"
                    yield SampleSpec(
                        observation=triplet['Observation'],
                        target_action=triplet['Action'],
                        feedback={"text": triplet['Feedback']},
                        focus_key=(shuffle_idx, triplet['fb_idx']),  # shuffle_idx, step_id
                        memory_directives={"memory_snapshot": self._get_memory_snapshot(selected)},
                        meta={
                            "triple_index": triplet['fb_idx'],
                            "shuffle_index": shuffle_idx,
                            "subset_size": len(selected),
                            # If dataset is provided, use triplet order as env_index for evaluation
                            "env_index": i if hasattr(self, "target_environments") else None,
                        },
                        sample_id=sample_id,
                    )
            else:
                selected = sampled_rest.copy()
                if self.shuffle_triplets:
                    random.shuffle(selected)
                sample_id = f"triplet_{triplet['fb_idx']}_shuffle_0"
                yield SampleSpec(
                    observation=triplet['Observation'],
                    target_action=triplet['Action'],
                    feedback={"text": triplet['Feedback']},
                    focus_key=(0, triplet['fb_idx']),  # episode/step indexing
                    memory_directives={"memory_snapshot": self._get_memory_snapshot(selected)},
                    meta={
                        "triple_end_index": triplet['fb_idx'],
                        "subset_size": len(selected),
                        "env_index": i if hasattr(self, "target_environments") else None,
                    },
                    sample_id=sample_id,
                )

    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        return spec.memory_directives["memory_snapshot"]


