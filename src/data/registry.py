from typing import Dict, Tuple, Type

from src.data.env import EnvDataset
from src.data.qa_dataset import QAEnvDataset, QAEnvDatasetConfig
from src.data.envs.omega_math_env import OmegaMathEnvDataset, OmegaMathEnvDatasetConfig
from src.data.envs.alfworld_env import ALFWorldEnvDataset, ALFWorldEnvDatasetConfig
from src.data.envs.finer_env import FinerEnvDataset, FinerEnvDatasetConfig
from src.data.envs.appworld_env import AppWorldEnvDataset, AppWorldEnvDatasetConfig


# Map dataset type key -> (ConfigCls, DatasetCls)
DATASET_REGISTRY: Dict[str, Tuple[Type, Type[EnvDataset]]] = {
    "qa": (QAEnvDatasetConfig, QAEnvDataset),
    "omega": (OmegaMathEnvDatasetConfig, OmegaMathEnvDataset),
    "alfworld": (ALFWorldEnvDatasetConfig, ALFWorldEnvDataset),
    "finer": (FinerEnvDatasetConfig, FinerEnvDataset),
    "appworld": (AppWorldEnvDatasetConfig, AppWorldEnvDataset),
}


