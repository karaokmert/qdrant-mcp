# Qdrant MCP Server

A Model Context Protocol (MCP) server that provides semantic memory capabilities using Qdrant vector database with configurable embedding providers.

## Features

- **Dynamic Collection Management**: Work with multiple collections from a single server instance
- **Multiple Embedding Providers**:
  - OpenAI (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
  - Sentence Transformers (all-MiniLM-L6-v2, all-mpnet-base-v2, and more)
- **Semantic Search**: Store and retrieve information using vector similarity
- **Flexible Configuration**: Environment variables for all settings
- **MCP Tools**: Store, find, delete, and list operations with collection support
- **Metadata Support**: Attach custom metadata to stored content

## Installation

### Via uvx (Recommended for MCP)

The server is designed to be lightweight by default. When using OpenAI embeddings:

```bash
# For OpenAI embeddings (lightweight, no ML dependencies)
uvx qdrant-mcp
```

For local embeddings with Sentence Transformers:

```bash
# For local embeddings (includes torch and other ML libraries)
uvx --with sentence-transformers qdrant-mcp
```

### Via pip (Development)

```bash
# Clone the repository
git clone https://github.com/andrewlwn77/qdrant-mcp.git
cd qdrant-mcp

# Basic install (OpenAI embeddings only)
pip install -e .

# With local embeddings support
pip install -e . sentence-transformers
```

## Configuration

The server can be configured using environment variables:

### Required Environment Variables

- `EMBEDDING_PROVIDER`: Choose between `openai` or `sentence-transformers`
- `EMBEDDING_MODEL`: Model name for the chosen provider
- `OPENAI_API_KEY`: Required when using OpenAI embeddings

### Optional Environment Variables

- `QDRANT_URL`: Qdrant server URL (default: `http://localhost:6333`)
- `QDRANT_API_KEY`: Qdrant API key (optional)
- `DEFAULT_COLLECTION_NAME`: Default Qdrant collection name (default: `mcp_memory`)
- `DEVICE`: Device for sentence transformers (default: auto-detect)
- `DEFAULT_LIMIT`: Default search results limit (default: 10)
- `SCORE_THRESHOLD`: Minimum similarity score (default: 0.0)

### Example Configuration

```bash
# OpenAI embeddings
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL=text-embedding-3-small
export OPENAI_API_KEY=your-api-key

# Sentence Transformers (local)
export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Supported Embedding Models

### OpenAI Models
- `text-embedding-3-small` (1536 dimensions) - Default
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions) - Legacy

### Sentence Transformers Models
- `all-MiniLM-L6-v2` (384 dimensions) - Fast and efficient
- `all-mpnet-base-v2` (768 dimensions) - Higher quality
- Any other Sentence Transformers model from Hugging Face

## Usage

### Dynamic Collections

The server supports working with multiple collections dynamically. You can:
- Store data in different collections by specifying `collection_name` in each request
- Query specific collections for targeted searches
- Manage multiple agents or contexts with separate memory spaces
- Use a single server instance for all collections (efficient resource usage)

Example workflow:
```python
# Store in agent-specific collections
qdrant_store("Clara's memory", collection_name="clara-agent")
qdrant_store("Assistant's data", collection_name="assistant-agent")

# Search in specific collections
qdrant_find("previous conversations", collection_name="clara-agent")

# Use default collection when not specified
qdrant_store("General information")  # Uses DEFAULT_COLLECTION_NAME
```

### Starting the Server

```bash
# Development mode
python -m qdrant_mcp.server

# With MCP CLI
mcp dev src/qdrant_mcp/server.py
```

### MCP Tools

All tools support an optional `collection_name` parameter for dynamic collection management. If not specified, the default collection will be used.

#### qdrant-store
Store content with semantic embeddings:
```json
{
  "content": "The capital of France is Paris",
  "metadata": "{\"category\": \"geography\", \"type\": \"fact\"}",
  "id": "optional-custom-id",
  "collection_name": "optional-collection-name"
}
```
Response includes the collection name: `"Stored successfully in collection 'geography' with ID: xxx"`

#### qdrant-find
Search for relevant information:
```json
{
  "query": "What is the capital of France?",
  "limit": 5,
  "filter": "{\"category\": \"geography\"}",
  "score_threshold": 0.7,
  "collection_name": "optional-collection-name"
}
```
Each result includes the collection name in the response.

#### qdrant-delete
Delete stored items:
```json
{
  "ids": "id1,id2,id3",
  "collection_name": "optional-collection-name"
}
```
Response includes the collection name.

#### qdrant-list-collections
List all collections in Qdrant:
```json
{}
```

#### qdrant-collection-info
Get information about a specific collection:
```json
{
  "collection_name": "optional-collection-name"
}
```
If collection_name is not provided, returns info about the default collection.

## Integration with Claude Desktop

Add to your Claude Desktop configuration:

### For OpenAI Embeddings (Lightweight)
```json
{
  "mcpServers": {
    "qdrant-memory": {
      "command": "uvx",
      "args": ["qdrant-mcp"],
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_API_KEY": "your-api-key",
        "QDRANT_URL": "https://your-instance.qdrant.io",
        "QDRANT_API_KEY": "your-qdrant-api-key"
      }
    }
  }
}
```

### For Local Embeddings (Sentence Transformers)
```json
{
  "mcpServers": {
    "qdrant-memory": {
      "command": "uvx",
      "args": ["--with", "sentence-transformers", "qdrant-mcp"],
      "env": {
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "QDRANT_URL": "https://your-instance.qdrant.io",
        "QDRANT_API_KEY": "your-qdrant-api-key"
      }
    }
  }
}
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

### Project Structure

```
qdrant-mcp/
├── src/
│   └── qdrant_mcp/
│       ├── __init__.py
│       ├── server.py           # MCP server implementation
│       ├── settings.py         # Configuration management
│       ├── qdrant_client.py    # Qdrant operations
│       └── embeddings/
│           ├── base.py         # Abstract base class
│           ├── factory.py      # Provider factory
│           ├── openai.py       # OpenAI implementation
│           └── sentence_transformers.py  # ST implementation
└── tests/
    └── test_server.py
```

## Docker Support

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "-m", "qdrant_mcp.server"]
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.