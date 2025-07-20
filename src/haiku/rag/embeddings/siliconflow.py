"""
SiliconFlow embedding provider for haiku.rag.

SiliconFlow provides high-quality embedding models through their API.
This module implements the embedder interface for SiliconFlow models.
"""
try:
    import httpx
    from haiku.rag.config import Config
    from haiku.rag.embeddings.base import EmbedderBase


    class Embedder(EmbedderBase):
        """SiliconFlow embedder implementation."""

        _model: str = Config.EMBEDDINGS_MODEL
        _vector_dim: int = Config.EMBEDDINGS_VECTOR_DIM

        def __init__(self, model: str, vector_dim: int):
            super().__init__(model, vector_dim)
            self._api_key = Config.SILICONFLOW_API_KEY
            self._base_url = Config.SILICONFLOW_BASE_URL

            if not self._api_key:
                raise ValueError("SILICONFLOW_API_KEY environment variable is required for SiliconFlow embeddings")

        async def embed(self, text: str) -> list[float]:
            """Generate embeddings using SiliconFlow API."""
            headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

            payload = {"model": self._model, "input": text, "encoding_format": "float"}

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(f"{self._base_url}/embeddings", headers=headers, json=payload,
                        timeout=30.0)
                    response.raise_for_status()

                    data = response.json()

                    if "data" not in data or not data["data"]:
                        raise ValueError("Invalid response format from SiliconFlow API")

                    embedding = data["data"][0]["embedding"]

                    if len(embedding) != self._vector_dim:
                        raise ValueError(f"Expected embedding dimension {self._vector_dim}, "
                                         f"got {len(embedding)} from model {self._model}")

                    return embedding

                except httpx.HTTPStatusError as e:
                    error_detail = ""
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get("error", {}).get("message", str(e))
                    except Exception:
                        error_detail = str(e)

                    raise RuntimeError(
                        f"SiliconFlow API error (status {e.response.status_code}): {error_detail}") from e

                except httpx.RequestError as e:
                    raise RuntimeError(f"SiliconFlow API request failed: {e}") from e

                except Exception as e:
                    raise RuntimeError(f"Unexpected error in SiliconFlow embeddings: {e}") from e

except ImportError:
    # httpx is not available, skip this provider
    pass
