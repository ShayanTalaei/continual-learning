from typing import Union, Dict, Any, Optional
from logging import Logger

from src.datagen.strategies.strategy import StrategyConfig, Strategy
from src.datagen.strategies.replay_strategy import ReplayStrategy, ReplayStrategyConfig

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
    raise ValueError(f"Strategy type {type} not supported.")