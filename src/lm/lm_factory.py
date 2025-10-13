from typing import Optional
from logging import Logger
from src.lm.language_model import LMConfig, LanguageModel
from src.lm.gemini_client import GeminiClient, GeminiConfig
try:
    from src.lm.tokasaurus_client import TokasaurusClient, TokasaurusConfig
except Exception:
    TokasaurusClient = None  # type: ignore
    TokasaurusConfig = None  # type: ignore


def get_lm_client(lm_config: LMConfig, logger: Optional[Logger] = None) -> LanguageModel:
    if "gemini" in lm_config.model:
        # Accept LMConfig or GeminiConfig; coerce if needed
        if not isinstance(lm_config, GeminiConfig):
            lm_config = GeminiConfig(**lm_config.model_dump())
        return GeminiClient(lm_config, logger=logger)
    if lm_config.model.startswith("toka:"):
        if TokasaurusClient is None or TokasaurusConfig is None:
            raise ValueError("Tokasaurus client not available. Ensure tokasaurus is installed and src/lm/tokasaurus_client.py exists.")
        model_id = lm_config.model.split(":", 1)[1]
        # Build TokasaurusConfig from base LMConfig plus parsed model id
        cfg_dict = lm_config.model_dump()
        cfg_dict["model"] = model_id
        toka_cfg = TokasaurusConfig(**cfg_dict)  
        return TokasaurusClient(toka_cfg, logger=logger)
    raise ValueError(f"Model {lm_config.model} not supported")