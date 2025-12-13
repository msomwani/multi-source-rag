# backend/app/embeddings.py
import os
from typing import Iterable, List
from openai import OpenAI

from .config import settings

class OpenAIEmbeddingsWrapper:
    def __init__(self, model: str | None = None):
        # Use model from settings unless explicitly provided
        self.model = model or settings.EMBEDDING_MODEL

        # Ensure API key is available (early fail)
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Please add it to your .env file.")

        # create the new OpenAI client (v1+)
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: Iterable[str]) -> List[list[float]]:
        texts = list(texts)
        if not texts:
            return []

        # Use the new client API
        # response.data is a list of objects, each has .embedding attribute
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
