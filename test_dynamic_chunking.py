#!/usr/bin/env python
"""
测试动态切块策略
Test Dynamic Chunking Strategy
"""

import asyncio
import sys
from pathlib import Path

# 添加源代码路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from haiku.rag.domains.financial.chunker import FinancialChunker


async def test_dynamic_chunking():
    """测试动态切块功能"""
    print("=== 测试动态切块策略 ===\n")
    
    # 创建动态切块器
    chunker = FinancialChunker(
        chunk_size=500,           # 目标大小
        chunk_overlap=100,        # 重叠
        min_chunk_size=300,       # 最小大小
        max_chunk_size=500,       # 最大大小
        chunk_size_variance=100,  # 浮动范围
        preserve_tables=True,
        extract_metadata=True
    )
    
    print(f"切块器配置:")
    print(f"  目标大小: {chunker.chunk_size} tokens")
    print(f"  最小大小: {chunker.min_chunk_size} tokens")
    print(f"  最大大小: {chunker.max_chunk_size} tokens")
    print(f"  浮动范围: {chunker.chunk_size_variance} tokens")
    print(f"  重叠大小: {chunker.chunk_overlap} tokens")
    print()
    
    # 测试文本
    sample_text = """
香港交易所上市公司公告
股份代號：0700
公司名稱：騰訊控股有限公司

主要交易
關於收購遊戲開發公司之公告

1. 交易背景
董事會欣然宣布，本公司已於2024年1月15日簽署協議，以總代價港幣50億元收購ABC遊戲開發有限公司（「目標公司」）100%股權。

此項收購將有助本公司擴展其在移動遊戲領域的業務，特別是在亞太地區的市場份額。目標公司擁有多款熱門手機遊戲，包括角色扮演遊戲和策略遊戲，用戶基礎超過500萬人。

2. 目標公司資料
目標公司為一家在新加坡註冊成立的私人有限公司，主要從事手機遊戲開發和發行業務。其主要資產包括：

- 5款已發布的手機遊戲
- 3款正在開發中的新遊戲
- 完整的遊戲開發團隊（約150人）
- 遊戲引擎和相關技術專利

3. 財務資料
根據目標公司截至2023年12月31日止年度的經審核財務報表：

收入：港幣8.5億元
毛利：港幣6.2億元
淨利潤：港幣2.1億元
總資產：港幣12.3億元
淨資產：港幣8.7億元

4. 交易條款
收購代價：港幣50億元
付款方式：現金支付
完成日期：預計2024年6月30日
先決條件：獲得相關監管機構批准

5. 對本公司的影響
此項收購預期將對本公司產生以下正面影響：

(a) 擴大遊戲業務規模
收購完成後，本公司將新增5款成熟的手機遊戲產品，預計將為本公司帶來額外的年收入約港幣8-10億元。

(b) 增強技術實力
目標公司的遊戲引擎技術將與本公司現有技術平台整合，提升整體研發能力。

(c) 拓展市場覆蓋
透過目標公司在東南亞市場的既有網絡，本公司可更有效地拓展該地區的業務。

6. 風險因素
董事會提醒股東注意以下風險：

- 遊戲行業競爭激烈，市場變化迅速
- 監管環境可能發生變化
- 整合過程可能面臨挑戰
- 匯率波動風險

7. 董事會意見
董事會認為此項收購符合本公司的長期發展策略，有助提升本公司在遊戲行業的競爭地位。董事會一致建議股東投票贊成此項交易。

8. 股東大會
本公司將於2024年3月15日召開股東特別大會，就此項收購進行表決。詳細資料將於稍後發出的通函中載明。

承董事會命
騰訊控股有限公司
主席
馬化騰

2024年1月20日
"""
    
    # 執行切块
    chunks = await chunker.chunk(sample_text)
    
    print(f"切块结果:")
    print(f"  总块数: {len(chunks)}")
    print()
    
    # 分析每个块
    for i, chunk in enumerate(chunks, 1):
        tokens = chunker.encoder.encode(chunk, disallowed_special=())
        token_count = len(tokens)
        
        print(f"块 {i}:")
        print(f"  Token数量: {token_count}")
        print(f"  字符数量: {len(chunk)}")
        print(f"  是否在范围内: {chunker.min_chunk_size <= token_count <= chunker.max_chunk_size}")
        
        # 显示块的开头
        preview = chunk[:100].replace('\n', ' ').strip()
        if len(chunk) > 100:
            preview += "..."
        print(f"  内容预览: {preview}")
        print()
    
    # 验证切块质量
    print("=== 切块质量分析 ===")
    
    total_tokens = sum(len(chunker.encoder.encode(chunk, disallowed_special=())) for chunk in chunks)
    original_tokens = len(chunker.encoder.encode(sample_text, disallowed_special=()))
    
    print(f"原文Token数: {original_tokens}")
    print(f"切块总Token数: {total_tokens}")
    print(f"重叠率: {((total_tokens - original_tokens) / original_tokens * 100):.1f}%")
    
    # 检查大小分布
    token_counts = [len(chunker.encoder.encode(chunk, disallowed_special=())) for chunk in chunks]
    avg_size = sum(token_counts) / len(token_counts)
    min_size = min(token_counts)
    max_size = max(token_counts)
    
    print(f"平均块大小: {avg_size:.1f} tokens")
    print(f"最小块大小: {min_size} tokens")
    print(f"最大块大小: {max_size} tokens")
    
    # 验证是否符合要求
    in_range_count = sum(1 for size in token_counts if chunker.min_chunk_size <= size <= chunker.max_chunk_size)
    print(f"符合范围的块: {in_range_count}/{len(chunks)} ({in_range_count/len(chunks)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(test_dynamic_chunking())
