import pytest
from datasets import Dataset

from haiku.rag.chunker import Chunker


@pytest.mark.asyncio
async def test_chunker(qa_corpus: Dataset):
    chunker = Chunker()
    doc = qa_corpus[0]["document_extracted"]
    chunks = await Chunker().chunk(doc)

    # Ensure that the text is split into multiple chunks
    assert len(chunks) > 1

    # Ensure that each chunk corresponds to roughly Config.CHUNK_SIZE tokens
    for chunk in chunks[:-1]:
        encoded_tokens = Chunker.encoder.encode(chunk, disallowed_special=())
        assert len(encoded_tokens) <= Chunker().chunk_size
        assert len(encoded_tokens) > Chunker().chunk_size * 0.9

    # Ensure that the last chunk is less than Config.CHUNK_SIZE tokens
    assert (
        len(Chunker.encoder.encode(chunks[-1], disallowed_special=()))
        < Chunker().chunk_size
    )

    # Test overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        current_tokens = Chunker.encoder.encode(current_chunk, disallowed_special=())
        next_tokens = Chunker.encoder.encode(next_chunk, disallowed_special=())

        overlap_size = min(chunker.chunk_overlap, len(current_tokens))
        current_overlap_tokens = current_tokens[-overlap_size:]
        next_overlap_tokens = next_tokens[:overlap_size]

        # The overlapping tokens should be identical
        assert current_overlap_tokens == next_overlap_tokens
        assert len(current_overlap_tokens) == min(
            chunker.chunk_overlap, len(current_tokens)
        )
