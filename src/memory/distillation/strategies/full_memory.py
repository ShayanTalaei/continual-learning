from typing import Iterable, List, Dict, Any, Optional
from pathlib import Path
import json
import random

from src.memory.history_list import Entry, HistoryList
from src.data.dataset_factory import build_dataset
from src.data.env import EnvDataset, Environment
from .base import MemoryFormationStrategy, SampleSpec


class FullMemoryStrategy(MemoryFormationStrategy):
    """Strategy that uses full memory from a checkpoint with samples from a target dataset.
    
    This strategy loads a complete memory snapshot from a checkpoint and creates samples
    using observations from a target dataset (e.g., validation set). Each sample uses
    the full memory as context.
    """
    
    def __init__(self, 
                 memory_checkpoint_path: str,
                 target_dataset_config: Dict[str, Any],
                 max_target_samples: Optional[int] = None,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None,
                 do_shuffle: bool = False,
                 num_shufflings: int = 1,
                 logger=None,
                 **kwargs: Any):
        """Initialize the FullMemoryStrategy.
        
        Args:
            memory_checkpoint_path: Path to the memory checkpoint directory
            target_dataset_config: Configuration for the target dataset (e.g., validation set)
            max_target_samples: Maximum number of samples to use from target dataset
            start_idx: Optional start index for slicing target dataset
            end_idx: Optional end index for slicing target dataset
            do_shuffle: Whether to shuffle memory entries for each sample
            num_shufflings: Number of shuffled versions to create per sample (when shuffling is enabled)
            logger: Optional logger instance
        """
        self.memory_checkpoint_path = Path(memory_checkpoint_path)
        self.target_dataset_config = target_dataset_config
        self.max_target_samples = max_target_samples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.do_shuffle = do_shuffle
        self.num_shufflings = num_shufflings
        self.logger = logger
        
        # Load the memory snapshot
        self._load_memory_snapshot()
        
        # Load the target dataset
        self._load_target_dataset()
    
    def _load_memory_snapshot(self):
        """Load the memory snapshot from the checkpoint directory."""
        # Find the latest memory snapshot
        memory_files = sorted(self.memory_checkpoint_path.glob("memory_*.jsonl"))
        if not memory_files:
            raise ValueError(f"No memory snapshots found in {self.memory_checkpoint_path}")
        
        latest_snapshot = memory_files[-1]
        if self.logger:
            self.logger.info(f"Loading memory snapshot from: {latest_snapshot}")
        
        self.memory_snapshot = HistoryList.load_snapshot(latest_snapshot)
        self.memory_entries = self.memory_snapshot.recall()
        
        if self.logger:
            self.logger.info(f"Loaded {len(self.memory_entries)} memory entries")
    
    def _load_target_dataset(self):
        """Load the target dataset."""
        if self.logger:
            self.logger.info(f"Loading target dataset with config: {self.target_dataset_config}")
        
        # Build the dataset using the dataset factory
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
            self.logger.info(f"Loaded {len(self.target_environments)} target environments")
    
    def _get_memory_triplets(self, memory_entries: List[Entry]) -> List[Dict[str, Any]]:
        """Organize memory entries into triplets (observation→action→feedback)."""
        triplets = []
        obs, act, fb = None, None, None
        
        for entry in memory_entries:
            if entry.type.lower() == "observation":
                obs = entry.content
                act, fb = None, None
            elif entry.type.lower() == "action":
                act = entry.content
            elif entry.type.lower() == "feedback":
                fb = entry.content
                if obs is not None and act is not None:
                    # We have a complete triplet
                    triplets.append({
                        'Observation': obs,
                        'Action': act,
                        'Feedback': fb
                    })
                obs, act, fb = None, None, None
        
        return triplets
    
    def _get_memory_snapshot_from_triplets(self, triplets: List[Dict[str, Any]]) -> List[Entry]:
        """Convert triplets back to memory entries."""
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
        """Generate samples using the full memory with target dataset observations.
        
        Note: The full_history parameter is ignored since we use our own memory snapshot.
        """
        # Convert memory entries to triplets for shuffling
        memory_triplets = self._get_memory_triplets(self.memory_entries)
        
        for env_idx, environment in enumerate(self.target_environments):
            # Reset the environment to get the initial observation
            obs = environment.reset()
            
            if self.do_shuffle:
                # Create multiple shuffled versions for this environment
                for shuffle_idx in range(self.num_shufflings):
                    # Create a shuffled copy of the triplets
                    shuffled_triplets = memory_triplets.copy()
                    random.shuffle(shuffled_triplets)
                    
                    # Convert back to memory entries
                    shuffled_memory = self._get_memory_snapshot_from_triplets(shuffled_triplets)
                    
                    sample_id = f"full_memory_env_{env_idx}_shuffle_{shuffle_idx}"
                    
                    yield SampleSpec(
                        observation=obs,
                        target_action=None,  # No target action for this strategy
                        feedback=None,  # No feedback for this strategy
                        focus_key=(shuffle_idx, env_idx),  # shuffle_idx, env_idx
                        memory_directives={"memory_snapshot": shuffled_memory},
                        meta={
                            "env_id": environment.env_id,
                            "env_type": environment.env_type,
                            "env_index": env_idx,
                            "shuffle_index": shuffle_idx,
                            "strategy": "full_memory"
                        },
                        sample_id=sample_id,
                    )
            else:
                # Use original memory without shuffling
                sample_id = f"full_memory_env_{env_idx}_shuffle_0"
                
                yield SampleSpec(
                    observation=obs,
                    target_action=None,  # No target action for this strategy
                    feedback=None,  # No feedback for this strategy
                    focus_key=(0, env_idx),  # episode_id=0, step_id=env_idx
                    memory_directives={"memory_snapshot": self.memory_entries},
                    meta={
                        "env_id": environment.env_id,
                        "env_type": environment.env_type,
                        "env_index": env_idx,
                        "strategy": "full_memory"
                    },
                    sample_id=sample_id,
                )
    
    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        """Return the full memory snapshot for each sample."""
        return spec.memory_directives["memory_snapshot"]
