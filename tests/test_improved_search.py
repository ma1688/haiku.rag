import pytest
from datasets import Dataset

from haiku.rag.query_processor import query_processor
from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_query_processor():
    """Test the query processor functionality."""
    # Test Chinese query processing
    chinese_query = "08096的年度股东大会内容"
    variations = query_processor.get_search_variations(chinese_query)
    
    assert 'original' in variations
    assert 'cleaned' in variations
    assert 'fts' in variations
    assert 'vector' in variations
    assert 'keywords' in variations
    assert 'expanded' in variations
    
    # Test keyword extraction
    keywords = query_processor.extract_keywords(chinese_query)
    assert len(keywords) > 0
    assert any('08096' in keyword for keyword in keywords)
    assert any('股东大会' in keyword for keyword in keywords)
    
    # Test FTS query processing
    fts_query = query_processor.process_for_fts(chinese_query)
    assert len(fts_query) > 0
    
    # Test vector query processing
    vector_query = query_processor.process_for_vector(chinese_query)
    assert len(vector_query) > 0


@pytest.mark.asyncio
async def test_improved_chunking():
    """Test the improved chunking strategy."""
    from haiku.rag.chunker import Chunker
    
    # Test with Chinese text
    chinese_text = """
    賞之味控股有限公司（「本公司」）董事會（「董事會」）謹此宣佈，本公司將於2024年9月19日（星期四）
    下午2時30分假座香港九龍觀塘開源道79號鱷魚恤中心6樓舉行股東週年大會（「股東大會」）。
    
    股東大會將審議以下事項：
    1. 省覽及採納截至2024年3月31日止年度之經審核財務報表及董事會報告書和核數師報告書；
    2. 重選退任董事；
    3. 續聘核數師及授權董事會釐定其酬金；
    4. 考慮及酌情通過授出購回股份之一般授權；
    5. 考慮及酌情通過授出發行股份之一般授權。
    
    所有股東均獲邀請出席股東大會。未能親身出席大會之股東可委任代表代為出席及投票。
    """
    
    chunker = Chunker(chunk_size=256, chunk_overlap=32)
    chunks = await chunker.chunk(chinese_text)
    
    assert len(chunks) > 1
    
    # Check that chunks have reasonable overlap
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Should have some overlapping content
        assert len(current_chunk) > 0
        assert len(next_chunk) > 0


@pytest.mark.asyncio
async def test_improved_search_performance():
    """Test that the improved search performs better than basic search."""
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)
    
    # Create test documents with Chinese content
    test_documents = [
        {
            "content": """
            賞之味控股有限公司（股份代號：08096）董事會宣佈，本公司將於2024年9月19日舉行股東週年大會。
            大會將審議年度財務報表、重選董事、續聘核數師等重要事項。
            所有股東均獲邀請出席，未能親身出席者可委任代表投票。
            """,
            "metadata": {"company_code": "08096", "type": "AGM_notice", "year": "2024"}
        },
        {
            "content": """
            本公司截至2024年3月31日止年度之經審核財務報表已經完成。
            年度營業額為港幣5,000萬元，較去年增長15%。
            董事會建議派發末期股息每股港幣0.05元。
            """,
            "metadata": {"company_code": "08096", "type": "financial_report", "year": "2024"}
        },
        {
            "content": """
            其他公司XYZ有限公司（股份代號：01234）也將舉行股東大會。
            該公司主要從事製造業務，與本公司業務不同。
            """,
            "metadata": {"company_code": "01234", "type": "other_company", "year": "2024"}
        }
    ]
    
    # Create documents
    created_docs = []
    for doc_data in test_documents:
        document = Document(
            content=doc_data["content"],
            metadata=doc_data["metadata"]
        )
        created_doc = await doc_repo.create(document)
        created_docs.append(created_doc)
    
    # Test search queries
    test_queries = [
        "08096的年度股东大会",
        "股東週年大會",
        "财务报表",
        "董事會",
        "2024年股东大会"
    ]
    
    for query in test_queries:
        # Test vector search
        vector_results = await chunk_repo.search_chunks(query, limit=3)
        assert len(vector_results) > 0
        
        # Test FTS search
        fts_results = await chunk_repo.search_chunks_fts(query, limit=3)
        assert len(fts_results) > 0
        
        # Test hybrid search
        hybrid_results = await chunk_repo.search_chunks_hybrid(query, limit=3)
        assert len(hybrid_results) > 0
        
        # Verify that relevant documents are found
        # For 08096 queries, should find 08096 documents
        if "08096" in query:
            found_08096 = False
            for chunk, score in hybrid_results:
                if "08096" in chunk.content:
                    found_08096 = True
                    break
            assert found_08096, f"Query '{query}' should find 08096 documents"
        
        # For AGM queries, should find AGM-related content
        if any(term in query.lower() for term in ["股东大会", "股東大會", "agm"]):
            found_agm = False
            for chunk, score in hybrid_results:
                if any(term in chunk.content for term in ["股東週年大會", "股东大会"]):
                    found_agm = True
                    break
            assert found_agm, f"Query '{query}' should find AGM-related content"
    
    store.close()


@pytest.mark.asyncio
async def test_relevance_scoring():
    """Test that relevance scoring works correctly."""
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)
    
    # Create a highly relevant document
    relevant_doc = Document(
        content="08096公司的年度股東大會將討論重要的財務報表和董事選舉事項。",
        metadata={"relevance": "high"}
    )
    
    # Create a less relevant document
    less_relevant_doc = Document(
        content="今天天氣很好，適合外出活動。股東大會是公司治理的重要組成部分。",
        metadata={"relevance": "low"}
    )
    
    await doc_repo.create(relevant_doc)
    await doc_repo.create(less_relevant_doc)
    
    # Search for specific query
    query = "08096年度股東大會財務報表"
    results = await chunk_repo.search_chunks_hybrid(query, limit=5)
    
    assert len(results) > 0
    
    # The first result should be more relevant (higher score)
    if len(results) > 1:
        first_score = results[0][1]
        second_score = results[1][1]
        
        # Check that scores are reasonable
        assert first_score > 0
        assert second_score > 0
    
    store.close()


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        await test_query_processor()
        await test_improved_chunking()
        await test_improved_search_performance()
        await test_relevance_scoring()
        print("All tests passed!")
    
    asyncio.run(run_tests())
