"""Qdrant client wrapper with embedding support."""

import uuid
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .embeddings import create_embedding_provider
from .settings import Settings


class QdrantMemoryClient:
    """Wrapper around Qdrant client with embedding support."""
    
    def __init__(self, settings: Settings):
        """Initialize Qdrant client with settings.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        # Parse URL to extract host and port
        from urllib.parse import urlparse
        parsed_url = urlparse(settings.qdrant_url)
        host = parsed_url.hostname
        port = parsed_url.port
        https = parsed_url.scheme == "https"
        
        # Use host/port parameters instead of URL for better compatibility
        if host:
            self.client = AsyncQdrantClient(
                host=host,
                port=port or (443 if https else 6333),
                https=https,
                api_key=settings.qdrant_api_key,
                timeout=30.0,  # Increased timeout for remote server
                prefer_grpc=False,  # Use REST API for better HTTPS support
            )
        else:
            # Fallback to URL parameter
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=30.0,  # Increased timeout for remote server
                prefer_grpc=False,  # Use REST API for better HTTPS support
            )
        
        # Create embedding provider
        self.embedding_provider = create_embedding_provider(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            api_key=settings.openai_api_key,
            device=settings.device,
        )
        
        # Track initialized collections
        self._initialized_collections = set()
    
    async def _ensure_collection(self, collection_name: str) -> None:
        """Ensure the collection exists (lazy initialization).
        
        Args:
            collection_name: Name of the collection to ensure exists
        """
        if collection_name in self._initialized_collections:
            return
            
        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                # Create collection with appropriate vector size
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_provider.dimensions,
                        distance=Distance.COSINE,
                ),
            )
            self._initialized_collections.add(collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant collection '{collection_name}': {e}")
    
    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
        collection_name: str | None = None
    ) -> dict[str, str]:
        """Store content with embeddings in Qdrant.
        
        Args:
            content: Text content to store
            metadata: Optional metadata to attach
            id: Optional ID for the point (generated if not provided)
            collection_name: Optional collection name (uses default if not provided)
            
        Returns:
            Dictionary with ID and collection name of the stored point
        """
        # Determine target collection
        target_collection = collection_name or self.settings.default_collection_name
        
        # Ensure collection exists
        await self._ensure_collection(target_collection)
        
        # Generate ID if not provided
        point_id = id or str(uuid.uuid4())
        
        # Generate embedding
        embedding = await self.embedding_provider.embed_text(content)
        
        # Prepare payload
        payload = {
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "embedding_model": self.embedding_provider.model_name,
            "embedding_provider": self.embedding_provider.provider_name,
        }
        
        if metadata:
            payload["metadata"] = metadata
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )
        
        # Upsert to Qdrant
        await self.client.upsert(
            collection_name=target_collection,
            points=[point],
        )
        
        return {
            "id": point_id,
            "collection": target_collection
        }
    
    async def find(
        self,
        query: str,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        collection_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Find similar content using semantic search.
        
        Args:
            query: Search query text
            limit: Number of results to return
            filter: Optional filter conditions
            score_threshold: Minimum score threshold
            collection_name: Optional collection name (uses default if not provided)
            
        Returns:
            List of search results with content and metadata
        """
        # Determine target collection
        target_collection = collection_name or self.settings.default_collection_name
        
        # Ensure collection exists
        await self._ensure_collection(target_collection)
        
        # Use defaults from settings if not provided
        limit = limit or self.settings.default_limit
        score_threshold = score_threshold or self.settings.score_threshold
        
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)
        
        # Build filter if provided
        search_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value),
                    )
                )
            if conditions:
                search_filter = Filter(must=conditions)
        
        # Search
        results = await self.client.search(
            collection_name=target_collection,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "timestamp": result.payload.get("timestamp", ""),
                "metadata": result.payload.get("metadata", {}),
                "embedding_model": result.payload.get("embedding_model", ""),
                "embedding_provider": result.payload.get("embedding_provider", ""),
                "collection": target_collection,
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    async def delete(self, ids: list[str], collection_name: str | None = None) -> dict[str, Any]:
        """Delete points by IDs.
        
        Args:
            ids: List of point IDs to delete
            collection_name: Optional collection name (uses default if not provided)
            
        Returns:
            Operation result
        """
        # Determine target collection
        target_collection = collection_name or self.settings.default_collection_name
        
        # Ensure collection exists
        await self._ensure_collection(target_collection)
        
        await self.client.delete(
            collection_name=target_collection,
            points_selector=ids,
        )
        
        return {
            "deleted": len(ids),
            "ids": ids,
            "collection": target_collection,
        }
    
    async def list_collections(self) -> list[str]:
        """List all collections in Qdrant.
        
        Returns:
            List of collection names
        """
        collections = await self.client.get_collections()
        return [c.name for c in collections.collections]
    
    async def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Get information about a collection.
        
        Args:
            collection_name: Optional collection name (uses default if not provided)
        
        Returns:
            Collection information
        """
        # Determine target collection
        target_collection = collection_name or self.settings.default_collection_name
        
        # Ensure collection exists
        await self._ensure_collection(target_collection)
        
        info = await self.client.get_collection(target_collection)
        return {
            "name": target_collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance,
        }
    
    async def close(self) -> None:
        """Close connections and cleanup."""
        if hasattr(self.embedding_provider, "close"):
            await self.embedding_provider.close()
        await self.client.close()