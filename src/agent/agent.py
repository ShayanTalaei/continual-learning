from typing import Optional, Generic, TypeVar
from logging import Logger, getLogger
from pydantic import BaseModel


class AgentConfig(BaseModel):
    pass


C = TypeVar("C", bound=AgentConfig)


class Agent(Generic[C]):
    def __init__(self, config: C, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("agent")

    def act(self, obs: str) -> str:
        raise NotImplementedError

    def observe(self, obs: Optional[str], feedback: dict, done: bool) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def end_episode(self) -> None:
        pass