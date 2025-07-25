# Haiku SQLite RAG

Retrieval-Augmented Generation (RAG) library on SQLite.

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work on SQLite alone without the need for external vector databases. It uses [sqlite-vec](https://github.com/asg017/sqlite-vec) for storing the embeddings and performs semantic (vector) search as well as full-text search combined through Reciprocal Rank Fusion. Both open-source (Ollama) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

## Features

- **Local SQLite**: No external servers required
- **Multiple embedding providers**: Ollama, VoyageAI, OpenAI, SiliconFlow
- **Multiple QA providers**: Ollama, OpenAI, Anthropic
- **Hybrid search**: Vector + full-text search with Reciprocal Rank Fusion
- **Question answering**: Built-in QA agents on your documents
- **File monitoring**: Auto-index files when run as server
- **40+ file formats**: PDF, DOCX, HTML, Markdown, audio, URLs
- **MCP server**: Expose as tools for AI assistants
- **CLI & Python API**: Use from command line or Python
- **Retrieval optimization**: Advanced tools for improving search performance
- **Chinese text support**: Optimized for Chinese documents and queries

## Quick Start

```bash
# Install
uv pip install haiku.rag

# Add documents
haiku-rag add "Your content here"
haiku-rag add-src document.pdf

# Search
haiku-rag search "query"

# Ask questions
haiku-rag ask "Who is the author of haiku.rag?"

# Rebuild database (re-chunk and re-embed all documents)
haiku-rag rebuild

# Optimize retrieval performance
python scripts/retrieval_optimizer.py database.db

# Start server with file monitoring
export MONITOR_DIRECTORIES="/path/to/docs"
haiku-rag serve
```

## Python Usage

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("database.db") as client:
    # Add document
    doc = await client.create_document("Your content")

    # Search
    results = await client.search("query")
    for chunk, score in results:
        print(f"{score:.3f}: {chunk.content}")

    # Ask questions
    answer = await client.ask("Who is the author of haiku.rag?")
    print(answer)
```

## MCP Server

Use with AI assistants like Claude Desktop:

```bash
haiku-rag serve --stdio
```

Provides tools for document management and search directly in your AI assistant.

## Documentation

Full documentation at: https://ggozad.github.io/haiku.rag/

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Provider setup
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - Environment variables
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
- [Retrieval Optimization](docs/retrieval_optimization.md) - Improve search performance
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) - Performance Benchmarks
