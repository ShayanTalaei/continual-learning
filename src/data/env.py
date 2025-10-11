from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
from logging import Logger, getLogger
from pydantic import BaseModel


class Environment(ABC):
    """Gym-style text environment protocol.

    reset() -> str
      Initialize episode and return first observation string.

    step(action: str) -> (next_obs, feedback, done, info)
      Advance the environment with the given action.
    """
    
    def __init__(self, env_id: str, env_type: str):
        self.env_id = env_id
        self.env_type = env_type

    @abstractmethod
    def reset(self) -> str:
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any]]:
        ...

    def evaluate(self, action: str) -> Dict[str, Any]:
        """Optional helper centralizing evaluation logic; step() may call this."""
        raise NotImplementedError


class EnvDatasetConfig(BaseModel):
    dataset_path: Optional[str] = None


class EnvDataset:
    def __init__(self, config: EnvDatasetConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("dataset")
        self.dataset: List[Environment] = self.load_dataset()

    def load_dataset(self) -> List[Environment]:
        return []

    def get_dataset(self) -> List[Environment]:
        return self.dataset