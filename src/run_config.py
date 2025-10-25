from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel

from .run_time import RunTimeConfig


class OutputConfig(BaseModel):
    results_dir: Optional[str] = None
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"


class RunConfig(BaseModel):
    runtime: RunTimeConfig
    train_dataset: Optional[Dict[str, Any]] = None
    validation_dataset: Optional[Dict[str, Any]] = None
    agent: Dict[str, Any]
    output: Optional[OutputConfig] = None
    seed: Optional[int] = None


