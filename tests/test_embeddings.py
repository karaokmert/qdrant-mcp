"""Tests for embedding providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from qdrant_mcp.embeddings import create_embedding_provider
from qdrant_mcp.embeddings.base import EmbeddingProvider
from qdrant_mcp.embeddings.openai import OpenAIEmbeddingProvider
from qdrant_mcp.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider


class TestEmbeddingFactory:
    """Test embedding provider factory."""
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_embedding_provider(
                provider="openai",
                model_name="text-embedding-3-small"
            )
            assert isinstance(provider, OpenAIEmbeddingProvider)
            assert provider.model_name == "text-embedding-3-small"
            assert provider.dimensions == 1536
    
    def test_create_sentence_transformers_provider(self):
        """Test creating Sentence Transformers provider."""
        with patch("sentence_transformers.SentenceTransformer"):
            provider = create_embedding_provider(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2"
            )
            assert isinstance(provider, SentenceTransformersEmbeddingProvider)
            assert provider.model_name == "all-MiniLM-L6-v2"
    
    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(
                provider="invalid",
                model_name="test"
            )


class TestOpenAIProvider:
    """Test OpenAI embedding provider."""
    
    @pytest.mark.asyncio
    async def test_embed_text(self):
        """Test embedding single text."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingProvider("text-embedding-3-small")
            
            # Mock the HTTP client
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
            }
            mock_response.raise_for_status = Mock()
            
            provider.client.post = AsyncMock(return_value=mock_response)
            
            result = await provider.embed_text("test text")
            
            assert result == [0.1, 0.2, 0.3]
            provider.client.post.assert_called_once()
    
    def test_invalid_model(self):
        """Test invalid model raises error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="Unknown OpenAI model"):
                OpenAIEmbeddingProvider("invalid-model")
    
    def test_missing_api_key(self):
        """Test missing API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not provided"):
                OpenAIEmbeddingProvider("text-embedding-3-small")