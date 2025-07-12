"""Qdrant client wrapper with embedding support."""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint,
)

from .embeddings import EmbeddingProvider, create_embedding_provider
from .settings import Settings


class QdrantMemoryClient:
    """Wrapper around Qdrant client with embedding support."""
    
    def __init__(self, settings: Settings):
        """Initialize Qdrant client with settings.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        # Create embedding provider
        self.embedding_provider = create_embedding_provider(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            api_key=settings.openai_api_key,
            device=settings.device,
        )
        
        # Initialize collection
        self._init_collection()
    
    def _init_collection(self) -> None:
        """Initialize or verify the collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.settings.collection_name not in collection_names:
            # Create collection with appropriate vector size
            self.client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_provider.dimensions,
                    distance=Distance.COSINE,
                ),
            )
    
    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """Store content with embeddings in Qdrant.
        
        Args:
            content: Text content to store
            metadata: Optional metadata to attach
            id: Optional ID for the point (generated if not provided)
            
        Returns:
            ID of the stored point
        """
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
        self.client.upsert(
            collection_name=self.settings.collection_name,
            points=[point],
        )
        
        return point_id
    
    async def find(
        self,
        query: str,
        limit: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find similar content using semantic search.
        
        Args:
            query: Search query text
            limit: Number of results to return
            filter: Optional filter conditions
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results with content and metadata
        """
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
        results = self.client.search(
            collection_name=self.settings.collection_name,
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
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    async def delete(self, ids: List[str]) -> Dict[str, Any]:
        """Delete points by IDs.
        
        Args:
            ids: List of point IDs to delete
            
        Returns:
            Operation result
        """
        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=ids,
        )
        
        return {
            "deleted": len(ids),
            "ids": ids,
        }
    
    def list_collections(self) -> List[str]:
        """List all collections in Qdrant.
        
        Returns:
            List of collection names
        """
        collections = self.client.get_collections().collections
        return [c.name for c in collections]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection.
        
        Returns:
            Collection information
        """
        info = self.client.get_collection(self.settings.collection_name)
        return {
            "name": self.settings.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance,
        }
    
    async def close(self) -> None:
        """Close connections and cleanup."""
        if hasattr(self.embedding_provider, "close"):
            await self.embedding_provider.close()
        self.client.close()