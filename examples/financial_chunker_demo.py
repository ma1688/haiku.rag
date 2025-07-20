#!/usr/bin/env python
"""
演示如何使用金融文档切块器处理港交所公告
Demo: Using Financial Chunker for HKEx Announcements
"""

import asyncio
import os
from pathlib import Path
import sys

# 添加源代码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.financial_chunker import FinancialChunker
from haiku.rag.config import Config


async def demo_basic_chunking():
    """基础切块演示"""
    print("=== 基础切块演示 ===\n")
    
    # 创建金融文档切块器
    chunker = FinancialChunker(
        chunk_size=800,  # 演示用较小的块
        chunk_overlap=200,
        preserve_tables=True,
        extract_metadata=True
    )
    
    # 示例港交所公告
    sample_text = """
香港交易所上市公司公告
股份代號：0700
公司名稱：騰訊控股有限公司

主要交易
關於收購遊戲開發公司之公告

1. 交易背景
董事會欣然宣布，本公司已於2024年1月15日簽署協議，以總代價港幣50億元收購ABC遊戲開發有限公司（「目標公司」）100%股權。

2. 交易詳情
2.1 收購價格
總代價為港幣50億元，分三期支付：
- 首期：簽約時支付40%（港幣20億元）
- 第二期：完成盡職調查後支付30%（港幣15億元）
- 尾款：交割完成後支付30%（港幣15億元）

2.2 目標公司財務資料
根據目標公司最近期經審核財務報表：
營業收入：港幣8億元
淨利潤：港幣2億元
總資產：港幣12億元

3. 收購理由
本次收購將增強本集團在遊戲開發領域的實力，預期將帶來協同效應。

4. 財務影響
預計本次收購將為本集團帶來：
- 年收入增加約港幣10億元
- EBITDA貢獻約港幣3億元
- 預期投資回報率18%

5. 股東批准
根據上市規則，本交易構成主要交易，需要股東批准。
"""
    
    # 執行切塊
    chunks, metadata = await chunker.chunk(sample_text, return_metadata=True)
    
    # 顯示元數據
    print("提取的元數據:")
    print(f"  股票代碼: {metadata.get('stock_code', 'N/A')}")
    print(f"  公司名稱: {metadata.get('company_name', 'N/A')}")
    print(f"  公告類型: {metadata.get('type', 'N/A')}")
    print()
    
    # 顯示切塊結果
    print(f"文檔被切分為 {len(chunks)} 個塊:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- 塊 {i} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print(f"長度: {len(chunk)} 字符")
        print()


async def demo_table_preservation():
    """表格保護演示"""
    print("\n=== 表格保護演示 ===\n")
    
    chunker = FinancialChunker(
        chunk_size=500,
        chunk_overlap=100,
        preserve_tables=True
    )
    
    # 包含表格的文本
    table_text = """
財務摘要

綜合損益表（百萬港元）
----------------------------------------
項目            2023年度    2022年度   變動
----------------------------------------
營業收入         125,000     112,000   +11.6%
營業成本         (87,500)    (78,400)  +11.6%
毛利              37,500      33,600   +11.6%
營業費用         (21,000)    (19,500)  +7.7%
營業利潤          16,500      14,100   +17.0%
財務收入           1,200       1,000   +20.0%
稅前利潤          17,700      15,100   +17.2%
所得稅           (4,425)     (3,775)  +17.2%
淨利潤           13,275      11,325   +17.2%
----------------------------------------

每股數據
每股盈利（港元）    3.52        3.01    +16.9%
每股股息（港元）    1.20        1.00    +20.0%
"""
    
    chunks = await chunker.chunk(table_text)
    
    print(f"包含表格的文檔被切分為 {len(chunks)} 個塊\n")
    
    # 檢查表格是否被保護
    for i, chunk in enumerate(chunks, 1):
        if "營業收入" in chunk and "淨利潤" in chunk:
            print(f"✅ 塊 {i}: 表格被完整保留")
        elif "營業收入" in chunk or "淨利潤" in chunk:
            print(f"⚠️  塊 {i}: 表格可能被分割")


async def demo_bilingual_handling():
    """中英文混合處理演示"""
    print("\n=== 中英文混合處理演示 ===\n")
    
    chunker = FinancialChunker(chunk_size=600, chunk_overlap=150)
    
    bilingual_text = """
關連交易公告
CONNECTED TRANSACTION ANNOUNCEMENT

一、交易概述 Transaction Overview
本公司擬向控股股東收購物業，作價港幣3億元。
The Company intends to acquire a property from its controlling shareholder for HK$300 million.

二、定價基準 Pricing Basis
交易價格乃參考獨立估值師的評估報告釐定。
The transaction price was determined with reference to a valuation report by an independent valuer.

三、董事會意見 Board Opinion
董事會認為交易條款公平合理，符合本公司及股東整體利益。
The Board considers that the terms are fair and reasonable and in the interests of the Company and shareholders as a whole.
"""
    
    chunks = await chunker.chunk(bilingual_text)
    
    print(f"雙語文檔被切分為 {len(chunks)} 個塊\n")
    
    for i, chunk in enumerate(chunks, 1):
        # 檢查雙語內容是否保持在一起
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in chunk)
        has_english = any('a' <= char.lower() <= 'z' for char in chunk)
        
        if has_chinese and has_english:
            print(f"✅ 塊 {i}: 包含中英文內容")
        else:
            print(f"   塊 {i}: 僅包含{'中文' if has_chinese else '英文'}內容")


async def demo_with_haiku_rag():
    """與 haiku.rag 集成演示"""
    print("\n=== 與 haiku.rag 集成演示 ===\n")
    
    # 設置環境變量啟用金融切塊器
    os.environ["USE_FINANCIAL_CHUNKER"] = "true"
    os.environ["FINANCIAL_CHUNK_SIZE"] = "1500"
    os.environ["FINANCIAL_CHUNK_OVERLAP"] = "400"
    
    print("已設置環境變量:")
    print(f"  USE_FINANCIAL_CHUNKER = {os.environ.get('USE_FINANCIAL_CHUNKER')}")
    print(f"  FINANCIAL_CHUNK_SIZE = {os.environ.get('FINANCIAL_CHUNK_SIZE')}")
    print(f"  FINANCIAL_CHUNK_OVERLAP = {os.environ.get('FINANCIAL_CHUNK_OVERLAP')}")
    print()
    
    # 重新加載配置
    from haiku.rag.config import Config
    config = Config()
    
    print("配置已更新:")
    print(f"  使用金融切塊器: {config.USE_FINANCIAL_CHUNKER}")
    print(f"  金融文檔塊大小: {config.FINANCIAL_CHUNK_SIZE}")
    print(f"  金融文檔重疊: {config.FINANCIAL_CHUNK_OVERLAP}")
    print()
    
    # 現在使用 haiku-rag 時會自動使用金融切塊器
    print("現在處理港交所公告時會自動使用金融切塊器！")
    print("示例命令:")
    print("  haiku-rag add-src /path/to/hkex_announcement.pdf")


async def main():
    """運行所有演示"""
    print("港交所公告金融文檔切塊器演示")
    print("=" * 50)
    
    # 運行各個演示
    await demo_basic_chunking()
    await demo_table_preservation()
    await demo_bilingual_handling()
    await demo_with_haiku_rag()
    
    print("\n演示完成！")
    print("\n使用建議:")
    print("1. 設置環境變量 USE_FINANCIAL_CHUNKER=true 啟用金融切塊器")
    print("2. 調整 FINANCIAL_CHUNK_SIZE 和 FINANCIAL_CHUNK_OVERLAP 以優化效果")
    print("3. 對於特別長的表格，考慮單獨處理或使用專門的表格解析工具")


if __name__ == "__main__":
    asyncio.run(main())