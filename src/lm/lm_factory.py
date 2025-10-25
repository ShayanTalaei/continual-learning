from typing import Optional, Dict, Any, Union
from logging import Logger
from src.lm.language_model import LMConfig, LanguageModel
from src.lm.gemini_client import GeminiClient, GeminiConfig
from src.lm.tokasaurus_client import TokasaurusClient, TokasaurusConfig
from src.lm.embedding_model import EmbeddingConfig, EmbeddingModel
from src.lm.google_embeddings_client import GoogleEmbeddingsClient, GoogleEmbeddingsConfig



def get_lm_client(lm_config: Union[LMConfig, Dict[str, Any]], logger: Optional[Logger] = None) -> LanguageModel:
    # Handle both dict and LMConfig inputs
    if isinstance(lm_config, dict):
        model = lm_config.get("model", "")
        cfg_dict = lm_config.copy()
    else:
        model = lm_config.model
        cfg_dict = lm_config.model_dump(exclude_unset=True)
    
    if "gemini" in model:
        # Accept LMConfig or GeminiConfig; coerce if needed
        if isinstance(lm_config, GeminiConfig):
            return GeminiClient(lm_config, logger=logger)
        else:
            gemini_cfg = GeminiConfig(**cfg_dict)
            return GeminiClient(gemini_cfg, logger=logger)
    
    if model.startswith("toka:"):
        if TokasaurusClient is None or TokasaurusConfig is None:
            raise ValueError("Tokasaurus client not available. Ensure tokasaurus is installed and src/lm/tokasaurus_client.py exists.")
        model_id = model.split(":", 1)[1]
        # Build TokasaurusConfig from config dict plus parsed model id
        cfg_dict["model"] = model_id
        toka_cfg = TokasaurusConfig(**cfg_dict)  
        return TokasaurusClient(toka_cfg, logger=logger)
    
    raise ValueError(f"Model {model} not supported")


def get_embedding_client(cfg: Union[EmbeddingConfig, Dict[str, Any]], logger: Optional[Logger] = None) -> EmbeddingModel:
    # Accept either dict or EmbeddingConfig
    if isinstance(cfg, dict):
        model = cfg.get("model", "")
        cfg_dict = cfg.copy()
    else:
        model = cfg.model
        cfg_dict = cfg.model_dump(exclude_unset=True)

    # Currently support Google Vertex embeddings (text-embedding-* models)
    # Route any non-empty model to Google client for now
    if model:
        ge_cfg = GoogleEmbeddingsConfig(**cfg_dict)
        return GoogleEmbeddingsClient(ge_cfg, logger=logger)

    raise ValueError("Embedding model not supported or model id missing")