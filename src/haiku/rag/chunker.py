import re
from typing import ClassVar

import tiktoken

from haiku.rag.config import Config


class Chunker:
    """A class that chunks text into smaller pieces for embedding and retrieval.

    Args:
        chunk_size: The maximum size of a chunk in tokens.
        chunk_overlap: The number of tokens of overlap between chunks.
    """

    encoder: ClassVar[tiktoken.Encoding] = tiktoken.encoding_for_model("gpt-4o")

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP, ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better chunking, especially for Chinese documents."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Add sentence boundaries for Chinese text
        # Chinese sentence endings
        text = re.sub(r'([。！？；])', r'\1\n', text)

        # English sentence endings
        text = re.sub(r'([.!?;])\s+', r'\1\n', text)

        # Paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def _find_best_split_point(self, text: str, target_pos: int) -> int:
        """Find the best position to split text, preferring sentence boundaries."""
        # Look for sentence boundaries near the target position
        search_window = min(100, len(text) // 4)  # Search within 100 chars or 25% of text

        start_search = max(0, target_pos - search_window)
        end_search = min(len(text), target_pos + search_window)

        # Chinese sentence endings
        chinese_endings = ['。', '！', '？', '；']
        # English sentence endings
        english_endings = ['. ', '! ', '? ', '; ']

        best_pos = target_pos
        best_score = 0

        for i in range(start_search, end_search):
            score = 0

            # Check for Chinese sentence endings
            if i < len(text) and text[i] in chinese_endings:
                score = 10
            # Check for English sentence endings
            elif i < len(text) - 1 and text[i:i + 2] in english_endings:
                score = 10
            # Check for paragraph breaks
            elif i < len(text) - 1 and text[i:i + 2] == '\n\n':
                score = 8
            # Check for line breaks
            elif i < len(text) and text[i] == '\n':
                score = 5
            # Check for commas (lower priority)
            elif i < len(text) and text[i] in ['，', ',']:
                score = 2

            # Prefer positions closer to target
            distance_penalty = abs(i - target_pos) / search_window
            final_score = score * (1 - distance_penalty)

            if final_score > best_score:
                best_score = final_score
                best_pos = i + 1 if score > 0 else i

        return best_pos

    async def chunk(self, text: str) -> list[str]:
        """Split the text into chunks based on token boundaries with smart splitting.

        Args:
            text: The text to be split into chunks.

        Returns:
            A list of text chunks with token-based boundaries and overlap.
        """
        if not text:
            return []

        # Preprocess text for better chunking
        text = self._preprocess_text(text)

        encoded_tokens = self.encoder.encode(text, disallowed_special=())

        if self.chunk_size > len(encoded_tokens):
            return [text]

        chunks = []
        i = 0

        while i < len(encoded_tokens):
            start_i = i
            end_i = min(i + self.chunk_size, len(encoded_tokens))

            # If this is not the last chunk, try to find a better split point
            if end_i < len(encoded_tokens):
                # Decode to find character position
                chunk_tokens = encoded_tokens[start_i:end_i]
                temp_text = self.encoder.decode(chunk_tokens)

                # Find better split point
                better_end = self._find_best_split_point(temp_text, len(temp_text))

                # Re-encode to get token position
                if better_end < len(temp_text):
                    better_text = temp_text[:better_end]
                    better_tokens = self.encoder.encode(better_text, disallowed_special=())
                    end_i = start_i + len(better_tokens)

            chunk_tokens = encoded_tokens[start_i:end_i]
            chunk_text = self.encoder.decode(chunk_tokens).strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)

            # Exit loop if this was the last possible chunk
            if end_i >= len(encoded_tokens):
                break

            # Move forward with overlap
            i += max(1, end_i - start_i - self.chunk_overlap)

        return chunks


chunker = Chunker()
