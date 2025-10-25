from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from logging import Logger, getLogger


class EmbeddingConfig(BaseModel):
    model: str
    log_calls: bool = False
    # Retry/backoff
    max_retries: int = 5
    starting_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 10.0
    # Embedding-specific
    output_dimensionality: Optional[int] = None
    task_type: Optional[str] = None  # e.g., "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
    normalize: bool = True


class EmbeddingModel:
    def __init__(self, config: EmbeddingConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("embedding_model")

    def embed_one(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        # Default naive implementation using embed_one
        return [self.embed_one(t) for t in texts]


