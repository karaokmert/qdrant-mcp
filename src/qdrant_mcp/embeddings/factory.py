"""Factory for creating embedding providers."""

from typing import Any

from .base import EmbeddingProvider
from .openai import OpenAIEmbeddingProvider
from .sentence_transformers import (
    IMPORT_ERROR_MSG,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    SentenceTransformersEmbeddingProvider,
)


def create_embedding_provider(
    provider: str,
    model_name: str,
    **kwargs: Any
) -> EmbeddingProvider:
    """Create an embedding provider instance.
    
    Args:
        provider: Provider name ("openai" or "sentence-transformers")
        model_name: Model name for the provider
        **kwargs: Additional provider-specific arguments
        
    Returns:
        EmbeddingProvider instance
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model_name=model_name,
            api_key=kwargs.get("api_key")
        )
    elif provider == "sentence-transformers" or provider == "sentence_transformers":
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(IMPORT_ERROR_MSG)
        return SentenceTransformersEmbeddingProvider(
            model_name=model_name,
            device=kwargs.get("device")
        )
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported providers: openai, sentence-transformers"
        )


def get_supported_models() -> dict[str, dict[str, Any]]:
    """Get information about all supported models.
    
    Returns:
        Dictionary with model information
    """
    return {
        "openai": {
            "text-embedding-3-small": {"dimensions": 1536, "default": True},
            "text-embedding-3-large": {"dimensions": 3072},
            "text-embedding-ada-002": {"dimensions": 1536, "legacy": True},
        },
        "sentence-transformers": {
            "all-MiniLM-L6-v2": {"dimensions": 384, "default": True},
            "all-mpnet-base-v2": {"dimensions": 768},
        }
    }