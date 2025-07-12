"""Sentence Transformers embeddings provider implementation."""

from typing import List, Optional, TYPE_CHECKING
from .base import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Flag to check if sentence-transformers is available
SENTENCE_TRANSFORMERS_AVAILABLE = False
IMPORT_ERROR_MSG = """
sentence-transformers is not installed. To use local embeddings, install it with:

    pip install sentence-transformers

Or if using uvx:

    uvx --with sentence-transformers qdrant-mcp
"""

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers embeddings provider for local embeddings."""
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """Initialize Sentence Transformers embedding provider.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on (cpu, cuda, etc.)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(IMPORT_ERROR_MSG)
        
        if model_name not in self.MODEL_DIMENSIONS:
            # Try to load it anyway - sentence transformers has many models
            # Get dimensions after loading
            self.model = SentenceTransformer(model_name, device=device)
            # Encode a dummy text to get dimensions
            dummy_embedding = self.model.encode("test", convert_to_numpy=True)
            dimensions = len(dummy_embedding)
        else:
            dimensions = self.MODEL_DIMENSIONS[model_name]
            self.model = SentenceTransformer(model_name, device=device)
        
        super().__init__(model_name, dimensions)
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text using Sentence Transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Convert to numpy array and then to list
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using Sentence Transformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Batch encode for efficiency
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Convert numpy array to list of lists
        return embeddings.tolist()
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "sentence-transformers"