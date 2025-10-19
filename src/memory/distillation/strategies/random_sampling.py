from typing import Iterable, List, Dict, Any, Optional
from pathlib import Path
import json
import random

from src.memory.history_list import Entry, HistoryList
from src.data.dataset_factory import build_dataset
from src.data.env import EnvDataset, Environment
from .base import MemoryFormationStrategy, SampleSpec


class RandomSamplingStrategy(MemoryFormationStrategy):
    """Strategy that randomly samples K trajectories from memory for each sample.
    
    This strategy loads a complete memory snapshot from a checkpoint and creates samples
    using observations from a target dataset. For each sample, it randomly selects K
    trajectories (triplets) from the full memory to use as context.
    """
    
    def __init__(self, 
                 memory_checkpoint_path: str,
                 target_dataset_config: Dict[str, Any],
                 subset_size: int,
                 num_subsets: int = 1,
                 max_target_samples: Optional[int] = None,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None,
                 logger=None,
                 **kwargs: Any):
        """Initialize the RandomSamplingStrategy.
        
        Args:
            memory_checkpoint_path: Path to the memory checkpoint directory
            target_dataset_config: Configuration for the target dataset (e.g., validation set)
            subset_size: Number of trajectories (triplets) to sample for each memory view
            num_subsets: Number of different random subsets to create per environment
            max_target_samples: Maximum number of samples to use from target dataset
            start_idx: Optional start index for slicing target dataset
            end_idx: Optional end index for slicing target dataset
            logger: Optional logger instance
        """
        self.memory_checkpoint_path = Path(memory_checkpoint_path)
        self.target_dataset_config = target_dataset_config
        self.subset_size = subset_size
        self.num_subsets = num_subsets
        self.max_target_samples = max_target_samples
        self.start_idx = start_idx
        self.end_idx = end_idx
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
    
    def _sample_random_trajectories(self, all_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Randomly sample subset_size trajectories from all available triplets."""
        if len(all_triplets) <= self.subset_size:
            # If we have fewer triplets than subset_size, use all of them
            return all_triplets.copy()
        
        # Randomly sample subset_size triplets
        return random.sample(all_triplets, self.subset_size)
    
    def iter_samples(self, full_history: List[Entry]) -> Iterable[SampleSpec]:
        """Generate samples using randomly sampled trajectories from memory.
        
        Note: The full_history parameter is ignored since we use our own memory snapshot.
        """
        # Convert memory entries to triplets for sampling
        all_memory_triplets = self._get_memory_triplets(self.memory_entries)
        
        if self.logger:
            self.logger.info(f"Available memory triplets: {len(all_memory_triplets)}")
            self.logger.info(f"Sampling {self.subset_size} triplets per subset, {self.num_subsets} subsets per environment")
        
        for env_idx, environment in enumerate(self.target_environments):
            # Reset the environment to get the initial observation
            obs = environment.reset()
            
            # Create num_subsets different random samples for this environment
            for subset_idx in range(self.num_subsets):
                # Randomly sample trajectories for this subset
                sampled_triplets = self._sample_random_trajectories(all_memory_triplets)
                
                # Convert back to memory entries
                sampled_memory = self._get_memory_snapshot_from_triplets(sampled_triplets)
                
                sample_id = f"random_sampling_env_{env_idx}_subset_{subset_idx}"
                
                yield SampleSpec(
                    observation=obs,
                    target_action=None,  # No target action for this strategy
                    feedback=None,  # No feedback for this strategy
                    focus_key=(subset_idx, env_idx),  # subset_idx, env_idx
                    memory_directives={"memory_snapshot": sampled_memory},
                    meta={
                        "env_id": environment.env_id,
                        "env_type": environment.env_type,
                        "env_index": env_idx,
                        "subset_index": subset_idx,
                        "subset_size": len(sampled_triplets),
                        "total_available_triplets": len(all_memory_triplets),
                        "strategy": "random_sampling"
                    },
                    sample_id=sample_id,
                )
    
    def build_memory_for_sample(self, full_history: List[Entry], spec: SampleSpec) -> List[Entry]:
        """Return the sampled memory snapshot for each sample."""
        return spec.memory_directives["memory_snapshot"]
