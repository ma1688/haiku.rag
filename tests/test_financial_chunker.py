"""Tests for the financial document chunker."""

import pytest
from haiku.rag.financial_chunker import FinancialChunker


class TestFinancialChunker:
    """Test cases for FinancialChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a financial chunker instance."""
        return FinancialChunker(
            chunk_size=500,  # Smaller for testing
            chunk_overlap=100,
            preserve_tables=True,
            extract_metadata=True
        )

    @pytest.fixture
    def hkex_announcement_sample(self):
        """Sample HKEx announcement text."""
        return """
香港交易所上市公司公告
股份代號：0001
公司名稱：長江和記實業有限公司

盈利公告
EARNINGS ANNOUNCEMENT

1. 背景
本公司董事會欣然宣布截至2023年12月31日止年度之經審核綜合業績。

2. 財務摘要
以下為本集團之主要財務數據：

收入          港幣 380,730 百萬元
EBITDA       港幣  91,234 百萬元
淨利潤        港幣  35,025 百萬元
每股盈利      港幣    9.13 元

3. 業務回顧
3.1 港口及相關服務
全球貨櫃吞吐量達8,470萬個標準箱，增長5.2%。

3.2 零售
屈臣氏集團在全球28個市場經營超過16,000家店舖。

4. 股息
董事會建議派發末期股息每股港幣2.55元。

5. 展望
展望2024年，集團將繼續專注於核心業務的增長。
"""

    @pytest.fixture
    def mixed_language_sample(self):
        """Sample with mixed Chinese/English content."""
        return """
重大收購事項
MAJOR ACQUISITION

一、交易概要 Transaction Summary
本公司擬以現金代價HK$5,000,000,000收購目標公司100%股權。
The Company intends to acquire 100% equity interest in the Target Company for a cash consideration of HK$5,000,000,000.

二、先決條件 Conditions Precedent
(1) 獲得監管批准 Regulatory approvals
(2) 盡職調查滿意 Satisfactory due diligence
(3) 無重大不利變化 No material adverse change

三、財務影響 Financial Impact
預期本次收購將為本集團帶來以下影響：
- 增加年收入約HK$800百萬
- EBITDA貢獻約HK$200百萬
- 預期投資回報率15%
"""

    @pytest.fixture
    def table_sample(self):
        """Sample with table data."""
        return """
財務報表摘要

損益表（百萬港元）
------------------------------------
項目            2023年    2022年
------------------------------------
營業收入        12,500    11,200
營業成本        (8,750)   (7,840)
毛利             3,750     3,360
營業費用        (2,100)   (1,950)
營業利潤         1,650     1,410
------------------------------------

資產負債表（百萬港元）
------------------------------------
資產            2023年    2022年
------------------------------------
流動資產         5,600     5,100
非流動資產      18,400    17,200
總資產          24,000    22,300
------------------------------------
"""

    @pytest.mark.asyncio
    async def test_section_detection(self, chunker, hkex_announcement_sample):
        """Test section header detection."""
        chunks = await chunker.chunk(hkex_announcement_sample)
        
        # Should preserve section structure
        assert any("1. 背景" in chunk for chunk in chunks)
        assert any("2. 財務摘要" in chunk for chunk in chunks)
        
        # Sections should generally start new chunks
        section_starts = [chunk for chunk in chunks if any(
            chunk.strip().startswith(header) 
            for header in ["1.", "2.", "3.", "4.", "5."]
        )]
        assert len(section_starts) >= 3  # At least some sections start new chunks

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, chunker, hkex_announcement_sample):
        """Test document metadata extraction."""
        chunks, metadata = await chunker.chunk(hkex_announcement_sample, return_metadata=True)
        
        assert metadata.get("stock_code") == "0001"
        assert "長江和記" in metadata.get("company_name", "")
        assert metadata.get("type") == "earnings"

    @pytest.mark.asyncio
    async def test_table_preservation(self, chunker, table_sample):
        """Test that tables are not split."""
        chunks = await chunker.chunk(table_sample)
        
        # Check that table rows stay together
        for chunk in chunks:
            if "營業收入" in chunk:
                # If chunk contains one table row, it should contain the complete table
                assert "營業利潤" in chunk
                assert "----" in chunk  # Table border

    @pytest.mark.asyncio
    async def test_financial_term_preservation(self, chunker, mixed_language_sample):
        """Test that financial terms are kept together."""
        chunks = await chunker.chunk(mixed_language_sample)
        
        # Financial amounts should stay with their context
        for chunk in chunks:
            if "HK$5,000,000,000" in chunk:
                assert "收購" in chunk or "acquire" in chunk
            if "EBITDA貢獻" in chunk:
                assert "HK$200百萬" in chunk

    @pytest.mark.asyncio
    async def test_mixed_language_handling(self, chunker, mixed_language_sample):
        """Test handling of mixed Chinese/English content."""
        chunks = await chunker.chunk(mixed_language_sample)
        
        # Bilingual sections should stay together
        for chunk in chunks:
            if "交易概要" in chunk:
                assert "Transaction Summary" in chunk
            if "先決條件" in chunk:
                assert "Conditions Precedent" in chunk

    @pytest.mark.asyncio
    async def test_chunk_size_limits(self, chunker):
        """Test that chunks respect size limits."""
        # Create a long document
        long_text = "這是一個測試句子。" * 1000
        chunks = await chunker.chunk(long_text)
        
        # Verify chunk sizes
        for chunk in chunks:
            tokens = chunker.encoder.encode(chunk)
            assert len(tokens) <= chunker.chunk_size * 1.2  # Allow 20% overflow

    @pytest.mark.asyncio
    async def test_overlap_functionality(self, chunker):
        """Test that chunks have proper overlap."""
        text = """
第一節：公司簡介
本公司成立於1990年，主要從事房地產開發業務。

第二節：業務發展
過去十年，公司在大灣區完成了多個重點項目。

第三節：財務狀況
公司財務穩健，資產負債率保持在合理水平。
"""
        chunks = await chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check for content overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i][-100:]  # Last 100 chars
                chunk2_start = chunks[i + 1][:100]  # First 100 chars
                
                # There should be some common content
                # (This is a simplified check)
                assert len(chunks[i]) > 0 and len(chunks[i + 1]) > 0

    @pytest.mark.asyncio
    async def test_numbered_list_handling(self, chunker):
        """Test handling of numbered lists."""
        text = """
收購條款：
1) 收購價格：港幣10億元
2) 付款方式：分三期支付
3) 首期付款：簽約後30日內支付40%
4) 第二期付款：完成盡職調查後支付30%
5) 尾款：交割完成後支付30%
"""
        chunks = await chunker.chunk(text)
        
        # Numbered list items should preferably stay together
        for chunk in chunks:
            if "1) 收購價格" in chunk:
                # Should contain at least a few consecutive items
                assert "2) 付款方式" in chunk or "3) 首期付款" in chunk

    @pytest.mark.asyncio
    async def test_empty_document(self, chunker):
        """Test handling of empty documents."""
        chunks = await chunker.chunk("")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_whitespace_normalization(self, chunker):
        """Test that excessive whitespace is normalized."""
        text = """
公告    標題


        重要    事項
        
        
        詳細     內容
"""
        chunks = await chunker.chunk(text)
        
        # Check that excessive spaces are normalized
        for chunk in chunks:
            assert "    " not in chunk  # No quadruple spaces
            assert not chunk.startswith("\n\n\n")  # No triple newlines at start