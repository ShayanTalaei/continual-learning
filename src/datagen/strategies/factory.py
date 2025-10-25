from typing import Union, Dict, Any, Optional
from logging import Logger

from src.datagen.strategies.strategy import StrategyConfig, Strategy
from src.datagen.strategies.replay_strategy import ReplayStrategy, ReplayStrategyConfig
from src.datagen.strategies.synthetic_gen_strategy import SyntheticGenStrategy, SyntheticGenStrategyConfig
from src.datagen.strategies.reflection_strategy import ReflectionStrategy, ReflectionStrategyConfig

def build_strategy(config: Union[StrategyConfig, Dict[str, Any]], logger: Optional[Logger] = None) -> Strategy:
    if isinstance(config, dict):
        type = config.get("type")
        cfg_dict = config.copy()
    else:
        type = config.type
        cfg_dict = config.model_dump(exclude_unset=True)
    
    if type == "replay":
        if isinstance(config, ReplayStrategyConfig):
            return ReplayStrategy(config, logger=logger)
        else:
            return ReplayStrategy(ReplayStrategyConfig(**cfg_dict), logger=logger)
    elif type == "synthetic_gen":
        if isinstance(config, SyntheticGenStrategyConfig):
            return SyntheticGenStrategy(config, logger=logger)
        else:
            return SyntheticGenStrategy(SyntheticGenStrategyConfig(**cfg_dict), logger=logger)
    elif type == "reflection":
        if isinstance(config, ReflectionStrategyConfig):
            return ReflectionStrategy(config, logger=logger)
        else:
            return ReflectionStrategy(ReflectionStrategyConfig(**cfg_dict), logger=logger)
    raise ValueError(f"Strategy type {type} not supported.")