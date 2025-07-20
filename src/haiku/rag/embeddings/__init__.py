from haiku.rag.config import Config
from haiku.rag.embeddings.base import EmbedderBase


def get_embedder() -> EmbedderBase:
    """
    Factory function to get the appropriate embedder based on the configuration.
    """

    if Config.EMBEDDINGS_PROVIDER == "ollama":
        try:
            from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder
        except ImportError:
            raise ImportError(
                "Ollama embedder requires the 'ollama' package. "
                "Please install it with: pip install ollama"
            )
        return OllamaEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    if Config.EMBEDDINGS_PROVIDER == "voyageai":
        try:
            from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder
        except ImportError:
            raise ImportError(
                "VoyageAI embedder requires the 'voyageai' package. "
                "Please install haiku.rag with the 'voyageai' extra:"
                "uv pip install haiku.rag --extra voyageai"
            )
        return VoyageAIEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    if Config.EMBEDDINGS_PROVIDER == "openai":
        try:
            from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder
        except ImportError:
            raise ImportError(
                "OpenAI embedder requires the 'openai' package. "
                "Please install haiku.rag with the 'openai' extra:"
                "uv pip install haiku.rag --extra openai"
            )
        return OpenAIEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    if Config.EMBEDDINGS_PROVIDER == "siliconflow":
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            raise ImportError(
                "SiliconFlow embedder requires the 'httpx' package. "
                "Please install it with: pip install httpx"
            )
        return SiliconFlowEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    raise ValueError(f"Unsupported embedding provider: {Config.EMBEDDINGS_PROVIDER}")
