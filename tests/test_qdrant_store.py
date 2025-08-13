"""Tests for qdrant_store metadata validation functionality."""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from qdrant_mcp.server import qdrant_store


class TestQdrantStoreMetadataValidation:
    """Test metadata validation in qdrant_store function."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        mock_client = AsyncMock()
        mock_client.store.return_value = {
            "id": "test-id-123", 
            "collection": "test_collection"
        }
        
        with patch('qdrant_mcp.server.qdrant_client', mock_client):
            yield mock_client

    @pytest.mark.asyncio
    async def test_metadata_dict_input(self, mock_qdrant_client):
        """Test metadata as Python dictionary."""
        metadata = {
            "type": "test",
            "date": "2025-08-13",
            "purpose": "validation_test"
        }
        
        result = await qdrant_store(
            content="Test content",
            metadata=metadata
        )
        
        assert "Stored successfully" in result
        assert "test-id-123" in result
        
        # Verify the client was called with the correct metadata
        mock_qdrant_client.store.assert_called_once_with(
            content="Test content",
            metadata=metadata,
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_metadata_string_input(self, mock_qdrant_client):
        """Test metadata as JSON string."""
        metadata = '{"type": "test", "date": "2025-08-13"}'
        expected_metadata = {"type": "test", "date": "2025-08-13"}
        
        result = await qdrant_store(
            content="Test content", 
            metadata=metadata
        )
        
        assert "Stored successfully" in result
        
        # Verify the client was called with parsed JSON
        mock_qdrant_client.store.assert_called_once_with(
            content="Test content",
            metadata=expected_metadata,
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_metadata_none(self, mock_qdrant_client):
        """Test with no metadata."""
        result = await qdrant_store(content="Test content", metadata=None)
        
        assert "Stored successfully" in result
        
        mock_qdrant_client.store.assert_called_once_with(
            content="Test content",
            metadata=None,
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_metadata_empty_dict(self, mock_qdrant_client):
        """Test with empty metadata dict."""
        result = await qdrant_store(content="Test content", metadata={})
        
        assert "Stored successfully" in result
        
        mock_qdrant_client.store.assert_called_once_with(
            content="Test content",
            metadata={},
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_metadata_invalid_json_string(self, mock_qdrant_client):
        """Test metadata with malformed JSON string."""
        metadata = '{"type": "test", "date":}'  # malformed JSON
        
        with pytest.raises(ValueError) as exc_info:
            await qdrant_store(content="Test", metadata=metadata)
        
        assert "Invalid JSON in metadata" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_metadata_non_serializable_dict(self, mock_qdrant_client):
        """Test metadata with non-serializable objects."""
        metadata = {"date": datetime.now()}  # datetime not JSON serializable
        
        with pytest.raises(ValueError) as exc_info:
            await qdrant_store(content="Test", metadata=metadata)
        
        assert "Invalid metadata dict - contains non-serializable objects" in str(exc_info.value)
        assert "JSON serializable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_metadata_invalid_type(self, mock_qdrant_client):
        """Test metadata with invalid type (not str or dict)."""
        metadata = 123  # Invalid type
        
        with pytest.raises(ValueError) as exc_info:
            await qdrant_store(content="Test", metadata=metadata)
        
        error_msg = str(exc_info.value)
        assert "Metadata format error" in error_msg
        assert "JSON string" in error_msg
        assert "Python dict" in error_msg

    @pytest.mark.asyncio
    async def test_complex_valid_metadata_dict(self, mock_qdrant_client):
        """Test complex but valid metadata dict."""
        metadata = {
            "user_id": "user123",
            "tags": ["important", "work"],
            "priority": 5,
            "completed": False,
            "nested": {
                "category": "documentation",
                "subcategory": "api"
            }
        }
        
        result = await qdrant_store(content="Complex test", metadata=metadata)
        
        assert "Stored successfully" in result
        mock_qdrant_client.store.assert_called_once_with(
            content="Complex test",
            metadata=metadata,
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_complex_valid_metadata_string(self, mock_qdrant_client):
        """Test complex but valid metadata as JSON string."""
        metadata_dict = {
            "user_id": "user123",
            "tags": ["important", "work"],
            "priority": 5,
            "completed": False
        }
        metadata_string = json.dumps(metadata_dict)
        
        result = await qdrant_store(content="Complex test", metadata=metadata_string)
        
        assert "Stored successfully" in result
        mock_qdrant_client.store.assert_called_once_with(
            content="Complex test",
            metadata=metadata_dict,
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, mock_qdrant_client):
        """Ensure existing string format still works (backward compatibility)."""
        metadata = '{"existing": "format", "version": "old"}'
        expected_metadata = {"existing": "format", "version": "old"}
        
        result = await qdrant_store(content="Backward compatibility test", metadata=metadata)
        
        assert "Stored successfully" in result
        mock_qdrant_client.store.assert_called_once_with(
            content="Backward compatibility test",
            metadata=expected_metadata,
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_empty_json_string(self, mock_qdrant_client):
        """Test with empty JSON string."""
        metadata = '{}'
        
        result = await qdrant_store(content="Empty JSON test", metadata=metadata)
        
        assert "Stored successfully" in result
        mock_qdrant_client.store.assert_called_once_with(
            content="Empty JSON test",
            metadata={},
            id=None,
            collection_name=None
        )

    @pytest.mark.asyncio
    async def test_qdrant_client_not_initialized(self):
        """Test behavior when Qdrant client is not initialized."""
        with patch('qdrant_mcp.server.qdrant_client', None):
            with pytest.raises(RuntimeError) as exc_info:
                await qdrant_store(content="Test")
            
            assert "Qdrant client not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_all_parameters_with_dict_metadata(self, mock_qdrant_client):
        """Test all parameters work together with dict metadata."""
        metadata = {"test": "all_params"}
        
        result = await qdrant_store(
            content="Full test",
            metadata=metadata,
            id="custom-id",
            collection_name="custom_collection"
        )
        
        assert "Stored successfully" in result
        mock_qdrant_client.store.assert_called_once_with(
            content="Full test",
            metadata=metadata,
            id="custom-id",
            collection_name="custom_collection"
        )

    @pytest.mark.asyncio
    async def test_all_parameters_with_string_metadata(self, mock_qdrant_client):
        """Test all parameters work together with string metadata."""
        metadata = '{"test": "all_params_string"}'
        expected_metadata = {"test": "all_params_string"}
        
        result = await qdrant_store(
            content="Full test",
            metadata=metadata,
            id="custom-id", 
            collection_name="custom_collection"
        )
        
        assert "Stored successfully" in result
        mock_qdrant_client.store.assert_called_once_with(
            content="Full test",
            metadata=expected_metadata,
            id="custom-id",
            collection_name="custom_collection"
        )