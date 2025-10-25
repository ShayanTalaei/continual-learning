import os
from typing import Optional, List

from dotenv import load_dotenv
from google.oauth2 import service_account
from logging import Logger

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from .embedding_model import EmbeddingConfig, EmbeddingModel

load_dotenv(override=True)


class GoogleEmbeddingsConfig(EmbeddingConfig):
    pass


class GoogleEmbeddingsClient(EmbeddingModel):
    def __init__(self, config: GoogleEmbeddingsConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        self._model: Optional[TextEmbeddingModel] = None

    @property
    def cfg(self) -> GoogleEmbeddingsConfig:
        return self.config  # type: ignore[return-value]

    def embed_one(self, text: str) -> List[float]:
        model = self._get_model()
        inputs = [TextEmbeddingInput(text, task_type=self.config.task_type) if self.config.task_type else TextEmbeddingInput(text)]
        embeddings = model.get_embeddings(inputs, output_dimensionality=self.config.output_dimensionality)
        vec = embeddings[0].values
        if self.config.normalize:
            return self._l2_normalize(vec)
        return list(vec)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        inputs = [TextEmbeddingInput(t, task_type=self.config.task_type) if self.config.task_type else TextEmbeddingInput(t) for t in texts]
        embeddings = model.get_embeddings(inputs, output_dimensionality=self.config.output_dimensionality)
        results = [e.values for e in embeddings]
        if self.config.normalize:
            return [self._l2_normalize(v) for v in results]
        return [list(v) for v in results]

    def _get_model(self) -> TextEmbeddingModel:
        if self._model is None:
            credentials_path = os.getenv("GCP_CREDENTIALS")
            project_id = os.getenv("GCP_PROJECT")
            region = os.getenv("GCP_REGION")
            creds = None
            if credentials_path:
                creds = service_account.Credentials.from_service_account_file(credentials_path)
            vertexai.init(project=project_id, location=region, credentials=creds)
            self._model = TextEmbeddingModel.from_pretrained(self.config.model)
        return self._model

    def _l2_normalize(self, vec: List[float]) -> List[float]:
        s = sum(v * v for v in vec) ** 0.5
        if s == 0:
            return list(vec)
        return [v / s for v in vec]


