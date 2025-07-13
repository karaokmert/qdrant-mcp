"""Base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model_name: str, dimensions: int):
        """Initialize the embedding provider.
        
        Args:
            model_name: Name of the embedding model
            dimensions: Dimension of the embeddings
        """
        self.model_name = model_name
        self.dimensions = dimensions
    
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the embedding provider."""
        pass
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dimensions": self.dimensions
        }