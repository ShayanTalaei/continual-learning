from src.lm.language_model import LMConfig, LanguageModel
from src.lm.gemini_client import GeminiClient, GeminiConfig


def get_lm_client(lm_config: LMConfig) -> LanguageModel:
    if "gemini" in lm_config.model:
        # Accept LMConfig or GeminiConfig; coerce if needed
        if not isinstance(lm_config, GeminiConfig):
            lm_config = GeminiConfig(**lm_config.model_dump())
        return GeminiClient(lm_config)
    raise ValueError(f"Model {lm_config.model} not supported")