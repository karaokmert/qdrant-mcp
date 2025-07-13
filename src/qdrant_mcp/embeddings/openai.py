"""OpenAI embeddings provider implementation."""

import os

import httpx

from .base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider using their API."""
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        """Initialize OpenAI embedding provider.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        if model_name not in self.MODEL_DIMENSIONS:
            raise ValueError(f"Unknown OpenAI model: {model_name}. Supported models: {list(self.MODEL_DIMENSIONS.keys())}")
        
        super().__init__(model_name, self.MODEL_DIMENSIONS[model_name])
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        response = await self.client.post(
            "/embeddings",
            json={
                "input": texts,
                "model": self.model_name,
                "encoding_format": "float"
            }
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Sort by index to ensure correct order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings]
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()