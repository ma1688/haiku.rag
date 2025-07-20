import json
import re

from haiku.rag.chunker import chunker
from haiku.rag.embeddings import get_embedder
from haiku.rag.query_processor import query_processor
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.repositories.base import BaseRepository


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk database operations."""

    def __init__(self, store):
        super().__init__(store)
        self.embedder = get_embedder()

    async def create(self, entity: Chunk, commit: bool = True) -> Chunk:
        """Create a chunk in the database."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (document_id, content, metadata)
            VALUES (:document_id, :content, :metadata)
            """,
            {
                "document_id": entity.document_id,
                "content": entity.content,
                "metadata": json.dumps(entity.metadata),
            },
        )

        entity.id = cursor.lastrowid

        # Generate and store embedding
        embedding = await self.embedder.embed(entity.content)
        serialized_embedding = self.store.serialize_embedding(embedding)
        cursor.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (:chunk_id, :embedding)
            """,
            {"chunk_id": entity.id, "embedding": serialized_embedding},
        )

        # Insert into FTS5 table for full-text search
        cursor.execute(
            """
            INSERT INTO chunks_fts(rowid, content)
            VALUES (:rowid, :content)
            """,
            {"rowid": entity.id, "content": entity.content},
        )

        if commit:
            self.store._connection.commit()
        return entity

    async def get_by_id(self, entity_id: int) -> Chunk | None:
        """Get a chunk by its ID."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            SELECT id, document_id, content, metadata
            FROM chunks WHERE id = :id
            """,
            {"id": entity_id},
        )

        row = cursor.fetchone()
        if row is None:
            return None

        chunk_id, document_id, content, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}

        return Chunk(
            id=chunk_id, document_id=document_id, content=content, metadata=metadata
        )

    async def update(self, entity: Chunk) -> Chunk:
        """Update an existing chunk."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")
        if entity.id is None:
            raise ValueError("Chunk ID is required for update")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            UPDATE chunks
            SET document_id = :document_id, content = :content, metadata = :metadata
            WHERE id = :id
            """,
            {
                "document_id": entity.document_id,
                "content": entity.content,
                "metadata": json.dumps(entity.metadata),
                "id": entity.id,
            },
        )

        # Regenerate and update embedding
        embedding = await self.embedder.embed(entity.content)
        serialized_embedding = self.store.serialize_embedding(embedding)
        cursor.execute(
            """
            UPDATE chunk_embeddings
            SET embedding = :embedding
            WHERE chunk_id = :chunk_id
            """,
            {"embedding": serialized_embedding, "chunk_id": entity.id},
        )

        # Update FTS5 table
        cursor.execute(
            """
            UPDATE chunks_fts
            SET content = :content
            WHERE rowid = :rowid
            """,
            {"content": entity.content, "rowid": entity.id},
        )

        self.store._connection.commit()
        return entity

    async def delete(self, entity_id: int, commit: bool = True) -> bool:
        """Delete a chunk by its ID."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Delete from FTS5 table first
        cursor.execute(
            "DELETE FROM chunks_fts WHERE rowid = :rowid", {"rowid": entity_id}
        )

        # Delete the embedding
        cursor.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = :chunk_id",
            {"chunk_id": entity_id},
        )

        # Delete the chunk
        cursor.execute("DELETE FROM chunks WHERE id = :id", {"id": entity_id})

        deleted = cursor.rowcount > 0
        if commit:
            self.store._connection.commit()
        return deleted

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Chunk]:
        """List all chunks with optional pagination."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        query = "SELECT id, document_id, content, metadata FROM chunks ORDER BY document_id, id"
        params = {}

        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        if offset is not None:
            query += " OFFSET :offset"
            params["offset"] = offset

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            Chunk(
                id=chunk_id,
                document_id=document_id,
                content=content,
                metadata=json.loads(metadata_json) if metadata_json else {},
            )
            for chunk_id, document_id, content, metadata_json in rows
        ]

    async def create_chunks_for_document(
        self, document_id: int, content: str, commit: bool = True
    ) -> list[Chunk]:
        """Create chunks and embeddings for a document."""
        # Chunk the document content
        chunk_texts = await chunker.chunk(content)
        created_chunks = []

        # Create chunks with embeddings using the create method
        for order, chunk_text in enumerate(chunk_texts):
            # Create chunk with order in metadata
            chunk = Chunk(
                document_id=document_id, content=chunk_text, metadata={"order": order}
            )

            created_chunk = await self.create(chunk, commit=commit)
            created_chunks.append(created_chunk)

        return created_chunks

    async def delete_all(self, commit: bool = True) -> bool:
        """Delete all chunks from the database."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("DELETE FROM chunk_embeddings")
        cursor.execute("DELETE FROM chunks")

        deleted = cursor.rowcount > 0
        if commit:
            self.store._connection.commit()
        return deleted

    async def delete_by_document_id(
        self, document_id: int, commit: bool = True
    ) -> bool:
        """Delete all chunks for a document."""
        chunks = await self.get_by_document_id(document_id)

        deleted_any = False
        for chunk in chunks:
            if chunk.id is not None:
                deleted = await self.delete(chunk.id, commit=False)
                deleted_any = deleted_any or deleted

        if commit and deleted_any and self.store._connection:
            self.store._connection.commit()
        return deleted_any

    async def search_chunks(
        self, query: str, limit: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Search for relevant chunks using vector similarity with query processing."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Process query for better vector search
        processed_query = query_processor.process_for_vector(query)

        # Generate embedding for the processed query
        query_embedding = await self.embedder.embed(processed_query)
        serialized_query_embedding = self.store.serialize_embedding(query_embedding)

        # Search for similar chunks using sqlite-vec with much expanded limit for better recall
        search_limit = min(limit * 10, 100)  # Search many more candidates initially
        cursor.execute(
            """
            SELECT c.id, c.document_id, c.content, c.metadata, distance, d.uri, d.metadata as document_metadata
            FROM chunk_embeddings
            JOIN chunks c ON c.id = chunk_embeddings.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE embedding MATCH :embedding AND k = :k
            ORDER BY distance
            """,
            {"embedding": serialized_query_embedding, "k": search_limit},
        )

        results = cursor.fetchall()

        # Apply relevance filtering and re-ranking
        filtered_results = []
        for chunk_id, document_id, content, metadata_json, distance, document_uri, document_metadata_json in results:
            # Calculate relevance score (lower distance = higher relevance)
            relevance_score = 1.0 / (1.0 + distance)

            # Apply aggressive keyword matching boost
            keywords = query_processor.extract_keywords(query)
            keyword_boost = 0.0
            content_lower = content.lower()
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # Higher boost for exact matches
                    keyword_boost += 0.3
                    # Extra boost for numbers (like stock codes)
                    if keyword.isdigit():
                        keyword_boost += 0.5
                # Partial matching for Chinese characters
                elif len(keyword) == 1 and '\u4e00' <= keyword <= '\u9fff':
                    if keyword in content:  # Case sensitive for Chinese
                        keyword_boost += 0.2

            final_score = relevance_score + keyword_boost

            filtered_results.append((
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    metadata=json.loads(metadata_json) if metadata_json else {},
                    document_uri=document_uri,
                    document_meta=json.loads(document_metadata_json)
                    if document_metadata_json
                    else {},
                ),
                final_score,
            ))

        # Sort by final score and return top results
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:limit]

    async def search_chunks_fts(
        self, query: str, limit: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Search for chunks using FTS5 full-text search with improved query processing."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Use improved query processing for FTS
        fts_query = query_processor.process_for_fts(query)

        # Expand search limit for much better recall
        search_limit = min(limit * 5, 50)

        # Search using FTS5 with multiple query strategies
        results = []

        # Strategy 1: Use processed FTS query
        try:
            cursor.execute(
                """
                SELECT c.id, c.document_id, c.content, c.metadata, rank, d.uri, d.metadata as document_metadata
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH :query
                ORDER BY rank
                LIMIT :limit
                """,
                {"query": fts_query, "limit": search_limit},
            )
            results.extend(cursor.fetchall())
        except Exception:
            # Fallback to simple keyword search if FTS query fails
            keywords = query_processor.extract_keywords(query)
            if keywords:
                simple_query = " OR ".join(keywords)
                cursor.execute(
                    """
                    SELECT c.id, c.document_id, c.content, c.metadata, rank, d.uri, d.metadata as document_metadata
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON c.document_id = d.id
                    WHERE chunks_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                    """,
                    {"query": simple_query, "limit": search_limit},
                )
                results.extend(cursor.fetchall())

        # Remove duplicates and process results
        seen_chunks = set()
        unique_results = []
        for result in results:
            chunk_id = result[0]
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)

        # Apply additional scoring and filtering
        scored_results = []
        for chunk_id, document_id, content, metadata_json, rank, document_uri, document_metadata_json in unique_results:
            # FTS5 rank is negative BM25 score, convert to positive
            base_score = -rank

            # Apply keyword matching boost
            keywords = query_processor.extract_keywords(query)
            keyword_boost = 0.0
            content_lower = content.lower()
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    keyword_boost += 0.2  # Higher boost for FTS as it's keyword-based

            final_score = base_score + keyword_boost

            scored_results.append((
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    metadata=json.loads(metadata_json) if metadata_json else {},
                    document_uri=document_uri,
                    document_meta=json.loads(document_metadata_json)
                    if document_metadata_json
                    else {},
                ),
                final_score,
            ))

        # Sort by final score and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:limit]

    async def search_chunks_hybrid(
        self, query: str, limit: int = 5, k: int = 60
    ) -> list[tuple[Chunk, float]]:
        """Enhanced hybrid search using improved RRF combining vector similarity and FTS5 full-text search."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Process query for different search methods
        query_variations = query_processor.get_search_variations(query)

        # Generate embedding for the processed vector query
        vector_query = query_variations['vector']
        query_embedding = await self.embedder.embed(vector_query)
        serialized_query_embedding = self.store.serialize_embedding(query_embedding)

        # Use processed FTS query
        fts_query = query_variations['fts']
        # Perform hybrid search using RRF (Reciprocal Rank Fusion)
        cursor.execute(
            """
            WITH vector_search AS (
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    c.metadata,
                    ROW_NUMBER() OVER (ORDER BY ce.distance) as vector_rank
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                WHERE ce.embedding MATCH :embedding AND k = :k_vector
                ORDER BY ce.distance
            ),
            fts_search AS (
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    c.metadata,
                    ROW_NUMBER() OVER (ORDER BY chunks_fts.rank) as fts_rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH :fts_query
                ORDER BY chunks_fts.rank
            ),
            all_chunks AS (
                SELECT id, document_id, content, metadata FROM vector_search
                UNION
                SELECT id, document_id, content, metadata FROM fts_search
            ),
            rrf_scores AS (
                SELECT
                    a.id,
                    a.document_id,
                    a.content,
                    a.metadata,
                    COALESCE(1.0 / (:k + v.vector_rank), 0) + COALESCE(1.0 / (:k + f.fts_rank), 0) as rrf_score
                FROM all_chunks a
                LEFT JOIN vector_search v ON a.id = v.id
                LEFT JOIN fts_search f ON a.id = f.id
            )
            SELECT r.id, r.document_id, r.content, r.metadata, r.rrf_score, d.uri, d.metadata as document_metadata
            FROM rrf_scores r
            JOIN documents d ON r.document_id = d.id
            ORDER BY r.rrf_score DESC
            LIMIT :limit
            """,
            {
                "embedding": serialized_query_embedding,
                "k_vector": limit * 3,
                "fts_query": fts_query,
                "k": k,
                "limit": limit,
            },
        )

        results = cursor.fetchall()
        return [
            (
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    metadata=json.loads(metadata_json) if metadata_json else {},
                    document_uri=document_uri,
                    document_meta=json.loads(document_metadata_json)
                    if document_metadata_json
                    else {},
                ),
                rrf_score,
            )
            for chunk_id, document_id, content, metadata_json, rrf_score, document_uri, document_metadata_json in results
        ]

    async def get_by_document_id(self, document_id: int) -> list[Chunk]:
        """Get all chunks for a specific document."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            SELECT c.id, c.document_id, c.content, c.metadata, d.uri, d.metadata as document_metadata
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = :document_id
            ORDER BY JSON_EXTRACT(c.metadata, '$.order')
            """,
            {"document_id": document_id},
        )

        rows = cursor.fetchall()
        return [
            Chunk(
                id=chunk_id,
                document_id=document_id,
                content=content,
                metadata=json.loads(metadata_json) if metadata_json else {},
                document_uri=document_uri,
                document_meta=json.loads(document_metadata_json)
                if document_metadata_json
                else {},
            )
            for chunk_id, document_id, content, metadata_json, document_uri, document_metadata_json in rows
        ]
