try:
    from voyageai.client import Client  # type: ignore

    from haiku.rag.config import Config
    from haiku.rag.embeddings.base import EmbedderBase

    class Embedder(EmbedderBase):
        _model: str = Config.EMBEDDINGS_MODEL
        _vector_dim: int = 1024

        async def embed(self, text: str) -> list[float]:
            client = Client()
            res = client.embed([text], model=self._model, output_dtype="float")
            return res.embeddings[0]  # type: ignore[return-value]

except ImportError:
    pass
