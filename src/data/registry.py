from typing import Dict, Tuple, Type

from src.data.env import EnvDataset
from src.data.qa_dataset import QAEnvDataset, QAEnvDatasetConfig
from src.data.envs.omega_math_env import OmegaMathEnvDataset, OmegaMathEnvDatasetConfig


# Map dataset type key -> (ConfigCls, DatasetCls)
DATASET_REGISTRY: Dict[str, Tuple[Type, Type[EnvDataset]]] = {
    "qa": (QAEnvDatasetConfig, QAEnvDataset),
    "omega": (OmegaMathEnvDatasetConfig, OmegaMathEnvDataset),
}


