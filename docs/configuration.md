# Configuration

Configuration is done through the use of environment variables.

!!! note
    If you create a db with certain settings and later change them, `haiku.rag` will detect incompatibilities (for example, if you change embedding provider) and will exit. You can **rebuild** the database to apply the new settings, see [Rebuild Database](./cli.md#rebuild-database).

## File Monitoring

Set directories to monitor for automatic indexing:

```bash
# Monitor single directory
MONITOR_DIRECTORIES="/path/to/documents"

# Monitor multiple directories
MONITOR_DIRECTORIES="/path/to/documents,/another_path/to/documents"
```

## Embedding Providers

If you use Ollama, you can use any pulled model that supports embeddings.

### Ollama (Default)

```bash
EMBEDDINGS_PROVIDER="ollama"
EMBEDDINGS_MODEL="mxbai-embed-large"
EMBEDDINGS_VECTOR_DIM=1024
```

### VoyageAI
If you want to use VoyageAI embeddings you will need to install `haiku.rag` with the VoyageAI extras,

```bash
uv pip install haiku.rag --extra voyageai
```

```bash
EMBEDDINGS_PROVIDER="voyageai"
EMBEDDINGS_MODEL="voyage-3.5"
EMBEDDINGS_VECTOR_DIM=1024
VOYAGE_API_KEY="your-api-key"
```

### OpenAI
If you want to use OpenAI embeddings you will need to install `haiku.rag` with the OpenAI extras,

```bash
uv pip install haiku.rag --extra openai
```

and set environment variables.

```bash
EMBEDDINGS_PROVIDER="openai"
EMBEDDINGS_MODEL="text-embedding-3-small"  # or text-embedding-3-large
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="your-api-key"
# Optional: for OpenAI-compatible APIs
OPENAI_BASE_URL="https://your-openai-compatible-api.com/v1"
```

### SiliconFlow
SiliconFlow provides high-quality embedding models. You need to install the `httpx` package:

```bash
pip install httpx
```

Then configure:

```bash
EMBEDDINGS_PROVIDER="siliconflow"
EMBEDDINGS_MODEL="Qwen/Qwen3-Embedding-8B"  # or other SiliconFlow models
EMBEDDINGS_VECTOR_DIM=4096  # depends on the model
SILICONFLOW_API_KEY="your-siliconflow-api-key"
SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"  # optional, this is the default
```

## Question Answering Providers

Configure which LLM provider to use for question answering.

### Ollama (Default)

```bash
QA_PROVIDER="ollama"
QA_MODEL="qwen3"
OLLAMA_BASE_URL="http://localhost:11434"
```

### OpenAI

For OpenAI QA, you need to install haiku.rag with OpenAI extras:

```bash
uv pip install haiku.rag --extra openai
```

Then configure:

```bash
QA_PROVIDER="openai"
QA_MODEL="gpt-4o-mini"  # or gpt-4, gpt-3.5-turbo, etc.
OPENAI_API_KEY="your-api-key"
# Optional: for OpenAI-compatible APIs
OPENAI_BASE_URL="https://your-openai-compatible-api.com/v1"
```

### Anthropic

For Anthropic QA, you need to install haiku.rag with Anthropic extras:

```bash
uv pip install haiku.rag --extra anthropic
```

Then configure:

```bash
QA_PROVIDER="anthropic"
QA_MODEL="claude-3-5-haiku-20241022"  # or claude-3-5-sonnet-20241022, etc.
ANTHROPIC_API_KEY="your-api-key"
```

## Mixed Provider Configurations

You can mix and match different providers for embeddings and QA. Here are some popular combinations:

### Example 1: OpenAI QA + SiliconFlow Embeddings

```bash
# QA using OpenAI (or OpenAI-compatible API)
QA_PROVIDER="openai"
QA_MODEL="gpt-4.1"
OPENAI_API_KEY="your-openai-api-key"
OPENAI_BASE_URL="https://api-0711-node144.be-a.dev/api/v1"

# Embeddings using SiliconFlow
EMBEDDINGS_PROVIDER="siliconflow"
EMBEDDINGS_MODEL="Qwen/Qwen3-Embedding-8B"
EMBEDDINGS_VECTOR_DIM=4096
SILICONFLOW_API_KEY="your-siliconflow-api-key"
```

### Example 2: Anthropic QA + VoyageAI Embeddings

```bash
# QA using Anthropic
QA_PROVIDER="anthropic"
QA_MODEL="claude-3-5-haiku-20241022"
ANTHROPIC_API_KEY="your-anthropic-api-key"

# Embeddings using VoyageAI
EMBEDDINGS_PROVIDER="voyageai"
EMBEDDINGS_MODEL="voyage-3.5"
EMBEDDINGS_VECTOR_DIM=1024
VOYAGE_API_KEY="your-voyage-api-key"
```

### Example 3: Local Ollama QA + Cloud Embeddings

```bash
# QA using local Ollama
QA_PROVIDER="ollama"
QA_MODEL="qwen3"
OLLAMA_BASE_URL="http://localhost:11434"

# Embeddings using OpenAI
EMBEDDINGS_PROVIDER="openai"
EMBEDDINGS_MODEL="text-embedding-3-small"
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="your-openai-api-key"
```

## Other Settings

### Database and Storage

```bash
# Default data directory (where SQLite database is stored)
DEFAULT_DATA_DIR="/path/to/data"
```

### Document Processing

```bash
# Chunk size for document processing
CHUNK_SIZE=256

# Chunk overlap for better context
CHUNK_OVERLAP=32
```
