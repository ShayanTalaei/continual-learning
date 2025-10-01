from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel

from src.run_time import RunTimeConfig


class OutputConfig(BaseModel):
    results_dir: Optional[str] = None
    save_memory_path: Optional[str] = None
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"


class RunConfig(BaseModel):
    runtime: RunTimeConfig
    dataset: Dict[str, Any]  # includes a `type` key to select dataset from registry
    agent: Dict[str, Any]
    output: Optional[OutputConfig] = None
    seed: Optional[int] = None


