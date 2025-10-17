from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel

from .run_time import RunTimeConfig


class OutputConfig(BaseModel):
    results_dir: Optional[str] = None
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"


class RunConfig(BaseModel):
    runtime: RunTimeConfig
    train_dataset: Dict[str, Any] | None = None
    validation_dataset: Dict[str, Any] | None = None
    agent: Dict[str, Any]
    output: Optional[OutputConfig] = None
    seed: Optional[int] = None


