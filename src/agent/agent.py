from typing import Optional, Generic, TypeVar
from logging import Logger, getLogger
from pydantic import BaseModel
from contextlib import contextmanager
from src.lm.language_model import LMConfig
from src.lm.lm_factory import get_lm_client
from src.lm.language_model import LanguageModel
from src.utils.logger import child


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