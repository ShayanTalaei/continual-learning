from typing import Any, Union
from pathlib import Path
from pydantic import BaseModel
from contextlib import contextmanager

class MemoryModuleConfig(BaseModel):
    _type: str


class MemoryModule:
    def __init__(self, config: MemoryModuleConfig):
        self.config = config
        self.training: bool = True

    def update(self, *args: Any, **kwargs: Any):
        if not self.training:
            return
        return self._update(*args, **kwargs)

    def _update(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def recall(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    @contextmanager
    def eval_mode(self):
        prev = self.training
        try:
            self.training = False
            yield self
        finally:
            self.training = prev

    # Snapshot API
    def save_snapshot(self, base_dir: Union[str, Path], snapshot_id: Union[int, str]) -> str:
        """Save a snapshot of this memory under base_dir and return the full path written.

        Implementations should choose filename/suffix and be idempotent.
        """
        raise NotImplementedError

    @classmethod
    def load_snapshot(cls, snapshot_path: Union[str, Path]) -> "MemoryModule":
        """Load a memory instance from snapshot file path and return it."""
        raise NotImplementedError