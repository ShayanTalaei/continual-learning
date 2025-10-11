from typing import Any, Dict, Optional

from logging import Logger, getLogger

from pydantic import BaseModel

from src.data.env import EnvDataset
from src.data.qa_dataset import QAEnvDataset, QAEnvDatasetConfig
from src.data.envs.omega_math_env import OmegaMathEnvDataset, OmegaMathEnvDatasetConfig
from src.data.registry import DATASET_REGISTRY


def _to_dict(conf: Any) -> Dict[str, Any]:
    if isinstance(conf, BaseModel):
        return conf.model_dump()
    if isinstance(conf, dict):
        return conf
    # Fallback: try to read __dict__
    return dict(conf.__dict__) if hasattr(conf, "__dict__") else {}


def build_dataset(conf: Any, logger: Optional[Logger] = None) -> EnvDataset:
    """Factory to build datasets from a generic config.

    Heuristics:
      - If config has key 'hf_dataset' (allenai omega schema), build OmegaMathEnvDataset
      - Else, default to QAEnvDataset
    """
    cfg_dict = _to_dict(conf)
    log = logger or getLogger("dataset_factory")

    # Registry-based construction when type is provided
    dtype = cfg_dict.get("type")
    if dtype:
        if dtype not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset type: {dtype}")
        ConfigCls, DatasetCls = DATASET_REGISTRY[dtype]
        ds_conf = ConfigCls(**cfg_dict)
        log.debug("Building dataset via registry: %s", dtype)
        return DatasetCls(ds_conf, logger=logger)

    # Heuristic fallback for omega datasets without explicit type
    if "hf_dataset" in cfg_dict and "omega" in str(cfg_dict["hf_dataset"]):
        ds_conf = OmegaMathEnvDatasetConfig(**cfg_dict)
        log.debug("Building OmegaMathEnvDataset for %s", ds_conf.hf_dataset)
        return OmegaMathEnvDataset(ds_conf, logger=logger)

    # Default: classic QA dataset config
    qa_conf = QAEnvDatasetConfig(**cfg_dict)
    log.debug("Building QAEnvDataset")
    return QAEnvDataset(qa_conf, logger=logger)


