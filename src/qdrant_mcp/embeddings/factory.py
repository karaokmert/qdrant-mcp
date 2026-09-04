"""Factory for creating embedding providers."""

from typing import Any

from .base import EmbeddingProvider
from .openai import OpenAIEmbeddingProvider
from .sentence_transformers import (
    IMPORT_ERROR_MSG,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    SentenceTransformersEmbeddingProvider,
)
from .tei import TEIEmbeddingProvider


def create_embedding_provider(
    provider: str,
    model_name: str,
    **kwargs: Any
) -> EmbeddingProvider:
    """Create an embedding provider instance.

    Args:
        provider: Provider name ("openai", "sentence-transformers" or "tei")
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
    elif provider == "tei":
        base_url = kwargs.get("tei_url")
        if not base_url:
            raise ValueError(
                "TEI provider requires TEI_URL (base URL of the TEI server)."
            )

        # e5 models require asymmetric prefixes; auto-enable them unless overridden.
        query_prefix = kwargs.get("query_prefix")
        document_prefix = kwargs.get("document_prefix")
        is_e5 = "e5" in model_name.lower()
        if query_prefix is None:
            query_prefix = "query: " if is_e5 else ""
        if document_prefix is None:
            document_prefix = "passage: " if is_e5 else ""

        return TEIEmbeddingProvider(
            model_name=model_name,
            base_url=base_url,
            api_key=kwargs.get("tei_api_key"),
            dimensions=kwargs.get("dimensions"),
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported providers: openai, sentence-transformers, tei"
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
        },
        "tei": {
            "intfloat/multilingual-e5-base": {"dimensions": 768, "default": True},
            "intfloat/multilingual-e5-large": {"dimensions": 1024},
            # Any model served by the TEI instance works; dimensions are
            # discovered automatically on first use.
        },
    }
