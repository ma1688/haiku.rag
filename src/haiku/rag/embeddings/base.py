class EmbedderBase:
    _model: str = ""
    _vector_dim: int = 0

    def __init__(self, model: str, vector_dim: int):
        self._model = model
        self._vector_dim = vector_dim

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embedder is an abstract class. Please implement the embed method in a subclass.")
