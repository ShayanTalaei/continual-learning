from typing import Optional
from logging import Logger
from src.lm.language_model import LMConfig, LanguageModel
from src.lm.gemini_client import GeminiClient, GeminiConfig
from src.lm.vllm_client import VLLMClient, VLLMConfig


def get_lm_client(lm_config: LMConfig, logger: Optional[Logger] = None) -> LanguageModel:
    model_str = lm_config.model

    if "gemini" in model_str:
        # Accept LMConfig or GeminiConfig; coerce if needed
        if not isinstance(lm_config, GeminiConfig):
            lm_config = GeminiConfig(**lm_config.model_dump())
        return GeminiClient(lm_config, logger=logger)

    if "vllm" in model_str:
        # Accept LMConfig or VLLMConfig; coerce if needed
        if not isinstance(lm_config, VLLMConfig):
            lm_config = VLLMConfig(**lm_config.model_dump())
        return VLLMClient(lm_config, logger=logger)

    raise ValueError(f"Model {lm_config.model} not supported")