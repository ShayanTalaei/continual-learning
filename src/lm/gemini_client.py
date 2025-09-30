import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from google.oauth2 import service_account

from .language_model import LMConfig, LanguageModel

load_dotenv(override=True)

class GeminiConfig(LMConfig):
    thinking_budget: Optional[int] = None

class GeminiClient(LanguageModel):
    """Synchronous Gemini client compatible with `LanguageModel` interface."""

    def __init__(self, config: GeminiConfig):
        super().__init__(config=config)
        self._gemini_client: Optional[genai.Client] = None

    def call(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        generate_content_config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.config.temperature ,
            max_output_tokens=self.config.max_output_tokens,
            thinking_config=(
                ThinkingConfig(thinking_budget=self.config.thinking_budget)
                if self.config.thinking_budget is not None
                else None
            ),
            # response_mime_type="application/json"
        )

        response = self._client().models.generate_content(
            model=self.config.model,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=generate_content_config,
        )
        
        return response.text

    def _client(self) -> genai.Client:
        if self._gemini_client is None:
            scopes = [
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/cloud-platform",
            ]

            credentials_path = os.getenv("GCP_CREDENTIALS")
            project_id = os.getenv("GCP_PROJECT")
            region = os.getenv("GCP_REGION")

            if not credentials_path:
                raise ValueError("GCP_CREDENTIALS environment variable not set")
            if not project_id:
                raise ValueError("GCP_PROJECT environment variable not set")
            if not region:
                raise ValueError("GCP_REGION environment variable not set")

            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )

            self._gemini_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
                credentials=credentials,
            )
        return self._gemini_client