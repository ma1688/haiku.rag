from ollama import AsyncClient

from haiku.rag.config import Config
from haiku.rag.embeddings.base import EmbedderBase


class Embedder(EmbedderBase):
    _model: str = Config.EMBEDDINGS_MODEL
    _vector_dim: int = 1024

    async def embed(self, text: str) -> list[float]:
        client = AsyncClient(host=Config.OLLAMA_BASE_URL)
        res = await client.embeddings(model=self._model, prompt=text)
        return list(res["embedding"])
