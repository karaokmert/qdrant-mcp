"""MCP server implementation for Qdrant."""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

from mcp import FastMCP
from mcp.types import (
    Tool,
    TextContent,
)

from .settings import get_settings
from .qdrant_client import QdrantMemoryClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings and MCP
settings = get_settings()
mcp = FastMCP(settings.server_name)

# Initialize Qdrant client (will be created on startup)
qdrant_client: Optional[QdrantMemoryClient] = None


@mcp.tool()
async def qdrant_store(content: str, metadata: Optional[str] = None, id: Optional[str] = None) -> str:
    """Store information in Qdrant with semantic embeddings.
    
    Args:
        content: The text content to store
        metadata: Optional JSON string with metadata
        id: Optional ID for the stored item
        
    Returns:
        ID of the stored item
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Parse metadata if provided
    metadata_dict = None
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise ValueError("Metadata must be valid JSON")
    
    # Store in Qdrant
    point_id = await qdrant_client.store(
        content=content,
        metadata=metadata_dict,
        id=id
    )
    
    return f"Stored successfully with ID: {point_id}"


@mcp.tool()
async def qdrant_find(
    query: str,
    limit: Optional[int] = None,
    filter: Optional[str] = None,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Find relevant information using semantic search.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        filter: Optional JSON string with filter conditions
        score_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of matching results with content and metadata
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Parse filter if provided
    filter_dict = None
    if filter:
        try:
            filter_dict = json.loads(filter)
        except json.JSONDecodeError:
            raise ValueError("Filter must be valid JSON")
    
    # Search in Qdrant
    results = await qdrant_client.find(
        query=query,
        limit=limit,
        filter=filter_dict,
        score_threshold=score_threshold
    )
    
    return results


@mcp.tool()
async def qdrant_delete(ids: str) -> Dict[str, Any]:
    """Delete items from Qdrant by their IDs.
    
    Args:
        ids: Comma-separated list of IDs to delete
        
    Returns:
        Deletion result
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Parse IDs
    id_list = [id.strip() for id in ids.split(",") if id.strip()]
    
    if not id_list:
        raise ValueError("No IDs provided")
    
    # Delete from Qdrant
    result = await qdrant_client.delete(id_list)
    
    return result


@mcp.tool()
async def qdrant_list_collections() -> List[str]:
    """List all collections in the Qdrant database.
    
    Returns:
        List of collection names
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    return qdrant_client.list_collections()


@mcp.tool()
async def qdrant_collection_info() -> Dict[str, Any]:
    """Get information about the current collection.
    
    Returns:
        Collection statistics and configuration
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    return qdrant_client.get_collection_info()


# Server lifecycle handlers
async def startup():
    """Initialize Qdrant client on server startup."""
    global qdrant_client
    try:
        qdrant_client = QdrantMemoryClient(settings)
        logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
        logger.info(f"Using collection: {settings.collection_name}")
        logger.info(f"Embedding provider: {settings.embedding_provider}")
        logger.info(f"Embedding model: {settings.embedding_model}")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise


async def shutdown():
    """Cleanup on server shutdown."""
    global qdrant_client
    if qdrant_client:
        await qdrant_client.close()
        logger.info("Qdrant client closed")


def main():
    """Main entry point for the MCP server."""
    # Register lifecycle handlers
    mcp.add_startup_handler(startup)
    mcp.add_shutdown_handler(shutdown)
    
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()