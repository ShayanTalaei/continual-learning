from typing import Optional, Generic, TypeVar
from logging import Logger, getLogger
from pydantic import BaseModel
from contextlib import contextmanager
from src.lm.language_model import LMConfig
from src.lm.lm_factory import get_lm_client
from src.lm.language_model import LanguageModel
from src.utils.logger import child
from src.utils import checkpoint as cputil
from pathlib import Path
import json


class AgentConfig(BaseModel):
    lm_config: LMConfig | None = None
    system_prompt: str | None = None
    verbose: bool = True


C = TypeVar("C", bound=AgentConfig)


class Agent(Generic[C]):
    def __init__(self, config: C, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("agent")
        self.lm: LanguageModel = get_lm_client(config.lm_config, logger=child(self.logger, "lm"))
        self.training: bool = True

    def act(self, obs: str) -> str:
        raise NotImplementedError

    def observe(self, obs: Optional[str], feedback: dict, done: bool) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def train(self) -> None:
        self.training = True
        mem = getattr(self, "memory", None)
        if mem is not None:
            try:
                mem.train()
            except Exception:
                pass

    def eval(self) -> None:
        self.training = False
        mem = getattr(self, "memory", None)
        if mem is not None:
            try:
                mem.eval()
            except Exception:
                pass

    @contextmanager
    def eval_mode(self):
        prev = self.training
        try:
            self.training = False
            yield self
        finally:
            self.training = prev

    def clone_for_episode(self, training: bool, share_memory: bool = True) -> "Agent":
        raise NotImplementedError

    # -----------------------------
    # Checkpointing (default hooks)
    # -----------------------------
    def save_checkpoint(self, checkpoint_dir: str, snapshot_id: int) -> dict:
        """Default checkpoint implementation.

        - Writes a minimal agent manifest.
        - If `self.memory` exists and supports `save_snapshot`, delegates to it and records the snapshot path.
        """
        ep_dir = Path(checkpoint_dir)
        ep_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "agent_type": self.__class__.__name__,
        }
        # Dump config shallowly if possible
        try:
            if hasattr(self, "config") and hasattr(self.config, "model_dump"):
                manifest["agent_config"] = self.config.model_dump()
        except Exception:
            pass

        memory_snapshot_path: Optional[str] = None
        mem = getattr(self, "memory", None)
        if mem is not None:
            try:
                memory_snapshot_path = mem.save_snapshot(ep_dir, snapshot_id)
                manifest["memory_snapshot_path"] = str(Path(memory_snapshot_path).name if memory_snapshot_path else None)
                # Also record memory class for clarity
                manifest["memory_type"] = mem.__class__.__name__
            except Exception:
                pass

        cputil.write_agent_manifest(ep_dir, manifest)
        return manifest

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Default restore implementation.

        - Reads agent manifest if present.
        - If `self.memory` exists, tries to load a snapshot written under the checkpoint directory.
        """
        ep_dir = Path(checkpoint_dir)
        manifest = cputil.read_agent_manifest(ep_dir)

        mem = getattr(self, "memory", None)
        if mem is None:
            return
        # Determine snapshot path
        snapshot_path: Optional[Path] = None
        try:
            rel = manifest.get("memory_snapshot_path") if isinstance(manifest, dict) else None
            if rel:
                cand = ep_dir / rel
                if cand.exists():
                    snapshot_path = cand
        except Exception:
            snapshot_path = None
        # Fallback: find any memory_* file
        if snapshot_path is None:
            for p in ep_dir.glob("memory_*"):
                snapshot_path = p
                break
        if snapshot_path is None:
            return
        # Use classmethod to load a new instance when possible
        try:
            loader = getattr(mem.__class__, "load_snapshot", None)
            if callable(loader):
                new_mem = loader(snapshot_path)
                setattr(self, "memory", new_mem)
        except Exception:
            pass