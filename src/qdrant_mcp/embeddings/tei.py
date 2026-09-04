"""Text Embeddings Inference (TEI) provider implementation.

Talks to a self-hosted Hugging Face TEI server (https://github.com/huggingface/text-embeddings-inference)
over its native /embed endpoint. Supports e5-style query/passage prefixes.
"""

import httpx

from .base import EmbeddingProvider


class TEIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by a self-hosted TEI server."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        """Initialize TEI embedding provider.

        Args:
            model_name: Name of the model served by TEI (informational, stored in payloads)
            base_url: Base URL of the TEI server (e.g. https://embed.example.com)
            api_key: Optional API key (sent as Bearer token)
            dimensions: Embedding dimensions; if None, discovered on first embed call
            query_prefix: Prefix prepended to search queries (e5 models: "query: ")
            document_prefix: Prefix prepended to stored documents (e5 models: "passage: ")
        """
        # Dimensions may be unknown until the first call; base class stores it as-is.
        super().__init__(model_name, dimensions)

        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=60.0,
        )

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string (no prefix)."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query, applying the query prefix."""
        return await self.embed_text(f"{self.query_prefix}{text}")

    async def embed_document(self, text: str) -> list[float]:
        """Embed a document for storage, applying the document prefix."""
        return await self.embed_text(f"{self.document_prefix}{text}")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using the TEI /embed endpoint."""
        if not texts:
            return []

        response = await self.client.post("/embed", json={"inputs": texts})
        response.raise_for_status()

        embeddings = response.json()

        # Discover dimensions on first successful call.
        if self.dimensions is None and embeddings:
            self.dimensions = len(embeddings[0])

        return embeddings

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "tei"

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
