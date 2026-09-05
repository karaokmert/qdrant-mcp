"""MCP server implementation for Qdrant."""

import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_mcp.qdrant_memory import QdrantMemoryClient
from qdrant_mcp.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Qdrant client (will be created on startup)
qdrant_client: QdrantMemoryClient | None = None


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncGenerator[None, None]:
    """Manage the lifecycle of the Qdrant client."""
    global qdrant_client
    try:
        # Startup
        settings = get_settings()
        qdrant_client = QdrantMemoryClient(settings)
        logger.info("Qdrant MCP server initialized")
        logger.info(f"Qdrant URL: {settings.qdrant_url}")
        logger.info(f"Default Collection: {settings.default_collection_name}")
        logger.info(f"Embedding: {settings.embedding_provider} / {settings.embedding_model}")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise
    finally:
        # Shutdown
        if qdrant_client:
            await qdrant_client.close()
            logger.info("Qdrant client closed")


# Initialize MCP with lifespan
mcp = FastMCP("qdrant-mcp", lifespan=lifespan)


@mcp.tool()
async def qdrant_store(content: str, metadata: str | dict[str, Any] | None = None, id: str | None = None, collection_name: str | None = None) -> str:
    """Store information in Qdrant with semantic embeddings.
    
    Args:
        content: The text content to store
        metadata: Optional metadata as JSON string or dict
        id: Optional ID for the stored item
        collection_name: Optional collection name (uses default if not provided)
        
    Returns:
        ID of the stored item
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Parse and validate metadata if provided
    metadata_dict = None
    if metadata is not None:
        try:
            if isinstance(metadata, dict):
                # Validate dict can be JSON serialized
                json.dumps(metadata)
                metadata_dict = metadata
            elif isinstance(metadata, str):
                # Validate and parse JSON string
                metadata_dict = json.loads(metadata)
            else:
                raise ValueError(
                    "Metadata format error. Please provide metadata as:\n"
                    "- JSON string: '{\"key\": \"value\"}'\n"
                    "- Python dict: {'key': 'value'}"
                )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metadata: {e}")
        except (TypeError, ValueError) as e:
            if "not JSON serializable" in str(e) or "Object of type" in str(e):
                raise ValueError(
                    f"Invalid metadata dict - contains non-serializable objects: {e}\n"
                    "Ensure all values are JSON serializable (str, int, float, bool, list, dict)"
                )
            raise ValueError(f"Invalid metadata: {e}")
    
    # Store in Qdrant
    result = await qdrant_client.store(
        content=content,
        metadata=metadata_dict,
        id=id,
        collection_name=collection_name
    )
    
    return f"Stored successfully in collection '{result['collection']}' with ID: {result['id']}"


@mcp.tool()
async def qdrant_find(
    query: str,
    limit: int | None = None,
    filter: str | dict[str, Any] | None = None,
    score_threshold: float | None = None,
    collection_name: str | None = None
) -> list[dict[str, Any]]:
    """Find relevant information using semantic search.

    Args:
        query: Search query text
        limit: Maximum number of results to return
        filter: Optional FLAT metadata filter as a dict or JSON string.
            Keys are metadata field names, values must match exactly.
            Example: {"tur": "kapanis"} or {"proje": "goat", "tur": "karar"}.
            Do NOT use Qdrant's native filter syntax ({"must": [...]}) here.
        score_threshold: Minimum similarity score (0-1)
        collection_name: Optional collection name (uses default if not provided)

    Returns:
        List of matching results with content and metadata
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")

    # Parse filter if provided (dict or JSON string)
    filter_dict = None
    if filter:
        if isinstance(filter, dict):
            filter_dict = filter
        else:
            try:
                filter_dict = json.loads(filter)
            except json.JSONDecodeError:
                raise ValueError("Filter must be valid JSON")
        if not isinstance(filter_dict, dict):
            raise ValueError("Filter must be a flat object like {\"tur\": \"kapanis\"}")
        if any(k in ("must", "should", "must_not") for k in filter_dict):
            raise ValueError(
                "Use a flat metadata filter like {\"tur\": \"kapanis\"} — "
                "Qdrant native filter syntax ({\"must\": [...]}) is not accepted here."
            )
    
    # Search in Qdrant
    results = await qdrant_client.find(
        query=query,
        limit=limit,
        filter=filter_dict,
        score_threshold=score_threshold,
        collection_name=collection_name
    )
    
    return results


@mcp.tool()
async def qdrant_get(ids: str, collection_name: str | None = None) -> list[dict[str, Any]]:
    """Retrieve records directly by their IDs (no semantic search).

    Use this to follow explicit links between records: when a record's
    metadata references related record IDs, fetch them with this tool.

    Args:
        ids: Comma-separated list of IDs to retrieve
        collection_name: Optional collection name (uses default if not provided)

    Returns:
        List of records with content and metadata (same shape as find, without score)
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")

    # Parse IDs
    id_list = [id.strip() for id in ids.split(",") if id.strip()]

    if not id_list:
        raise ValueError("No IDs provided")

    return await qdrant_client.get(id_list, collection_name=collection_name)


@mcp.tool()
async def qdrant_delete(ids: str, collection_name: str | None = None) -> dict[str, Any]:
    """Delete items from Qdrant by their IDs.
    
    Args:
        ids: Comma-separated list of IDs to delete
        collection_name: Optional collection name (uses default if not provided)
        
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
    result = await qdrant_client.delete(id_list, collection_name=collection_name)
    
    return result


@mcp.tool()
async def qdrant_list_collections() -> list[str]:
    """List all collections in the Qdrant database.
    
    Returns:
        List of collection names
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    return await qdrant_client.list_collections()


@mcp.tool()
async def qdrant_collection_info(collection_name: str | None = None) -> dict[str, Any]:
    """Get information about a collection.
    
    Args:
        collection_name: Optional collection name (uses default if not provided)
    
    Returns:
        Collection statistics and configuration
    """
    global qdrant_client
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    return await qdrant_client.get_collection_info(collection_name=collection_name)


def _bearer_guard(app, token: str):
    """Wrap ASGI app: reject requests without the expected bearer token."""
    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            if not hmac.compare_digest(auth, f"Bearer {token}"):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await app(scope, receive, send)
    return guarded


def main() -> None:
    """Main entry point for the MCP server."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        import uvicorn

        from mcp.server.transport_security import TransportSecuritySettings

        # Host dogrulamasi Traefik'te (Host-bazli route) + kapi bearer'da;
        # SDK'nin localhost-only varsayilani public domain'de 421 uretiyor.
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )

        app = mcp.streamable_http_app()
        token = os.environ.get("MCP_BEARER_TOKEN", "")
        if token:
            app = _bearer_guard(app, token)
        uvicorn.run(
            app,
            host=os.environ.get("MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_PORT", "8000")),
        )
    else:
        # Default: stdio (Claude Code and other local clients)
        mcp.run()


if __name__ == "__main__":
    main()