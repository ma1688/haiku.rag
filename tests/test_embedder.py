import numpy as np
import pytest

from haiku.rag.embeddings import get_embedder


@pytest.mark.asyncio
async def test_embedder():
    embedder = get_embedder()
    embedding = await embedder.embed("hello world")
    assert len(embedding) == embedder._vector_dim


@pytest.mark.asyncio
async def test_similarity():
    embedder = get_embedder()
    phrases = [
        "I enjoy eating great food.",
        "Python is my favorite programming language.",
        "I love to travel and see new places.",
    ]
    embeddings = [np.array(await embedder.embed(phrase)) for phrase in phrases]

    # Calculate cosine similarity
    def similarities(embeddings, test_embedding):
        return [
            np.dot(embedding, test_embedding)
            / (np.linalg.norm(embedding) * np.linalg.norm(test_embedding))
            for embedding in embeddings
        ]

    test_phrase = "I am going for a camping trip."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[2]

    test_phrase = "When is dinner ready?"
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[0]

    test_phrase = "I work as a software developer."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[1]


@pytest.mark.asyncio
async def test_openai_embedder(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

    try:
        from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder

        embedder = OpenAIEmbedder("text-embedding-3-small", 1536)

        # Mock the OpenAI client
        class MockEmbeddingData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embedding):
                self.data = [MockEmbeddingData(embedding)]

        class MockAsyncOpenAI:
            class MockEmbeddings:
                async def create(self, model, input):
                    return MockResponse([0.1] * 1536)

            def __init__(self):
                self.embeddings = self.MockEmbeddings()

        # Patch the AsyncOpenAI import
        import haiku.rag.embeddings.openai

        original_client = haiku.rag.embeddings.openai.AsyncOpenAI
        haiku.rag.embeddings.openai.AsyncOpenAI = MockAsyncOpenAI

        try:
            embedding = await embedder.embed("test text")
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
        finally:
            haiku.rag.embeddings.openai.AsyncOpenAI = original_client

    except ImportError:
        pytest.skip("OpenAI package not installed")


@pytest.mark.asyncio
async def test_voyageai_embedder(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "voyageai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "voyage-3.5")

    try:
        from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder

        embedder = VoyageAIEmbedder("voyage-3.5", 1024)

        # Mock the VoyageAI client
        class MockEmbeddings:
            def __init__(self, embeddings):
                self.embeddings = embeddings

        class MockClient:
            def embed(self, texts, model, output_dtype):
                return MockEmbeddings([[0.1] * 1024])

        # Patch the Client import
        import haiku.rag.embeddings.voyageai

        original_client = haiku.rag.embeddings.voyageai.Client
        haiku.rag.embeddings.voyageai.Client = MockClient

        try:
            embedding = await embedder.embed("test text")
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)
        finally:
            haiku.rag.embeddings.voyageai.Client = original_client

    except ImportError:
        pytest.skip("VoyageAI package not installed")
