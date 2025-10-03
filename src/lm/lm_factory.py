from typing import Optional
from logging import Logger
from src.lm.language_model import LMConfig, LanguageModel
from src.lm.gemini_client import GeminiClient, GeminiConfig


def get_lm_client(lm_config: LMConfig, logger: Optional[Logger] = None) -> LanguageModel:
    if "gemini" in lm_config.model:
        # Accept LMConfig or GeminiConfig; coerce if needed
        if not isinstance(lm_config, GeminiConfig):
            lm_config = GeminiConfig(**lm_config.model_dump())
        return GeminiClient(lm_config, logger=logger)
    raise ValueError(f"Model {lm_config.model} not supported")