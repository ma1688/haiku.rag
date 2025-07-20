#!/usr/bin/env python3
"""
简化的文本切片演示脚本

这个脚本专注于展示文本切片功能，不需要配置嵌入模型。
展示内容：
1. 原始文本
2. 切片后的文本块
3. 每个块的详细信息
4. 切片统计分析
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from haiku.rag.chunker import Chunker


def print_header(title: str, char: str = "=", width: int = 80):
    """打印标题"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_section(title: str, char: str = "-", width: int = 60):
    """打印章节标题"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


async def demo_chunking(text: str, chunk_size: int = 512, chunk_overlap: int = 64):
    """演示文本切片功能"""
    
    print_header("文本切片演示")
    
    # 创建切片器
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 1. 显示原始文本信息
    print_section("1. 原始文本分析")
    original_tokens = chunker.encoder.encode(text, disallowed_special=())
    
    print(f"文本长度: {len(text)} 字符")
    print(f"Token数量: {len(original_tokens)} tokens")
    print(f"切片配置: {chunk_size} tokens/块, {chunk_overlap} tokens重叠")
    
    print(f"\n原始文本内容:")
    print("-" * 40)
    print(text[:300] + "..." if len(text) > 300 else text)
    print("-" * 40)
    
    # 2. 执行切片
    print_section("2. 文本切片过程")
    chunks = await chunker.chunk(text)
    print(f"切片完成，共生成 {len(chunks)} 个文本块")
    
    # 3. 分析每个切片
    print_section("3. 切片详细分析")
    
    total_chunk_chars = 0
    total_chunk_tokens = 0
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = chunker.encoder.encode(chunk, disallowed_special=())
        chunk_char_len = len(chunk)
        chunk_token_len = len(chunk_tokens)
        
        total_chunk_chars += chunk_char_len
        total_chunk_tokens += chunk_token_len
        
        print(f"\n📄 切片 {i+1}:")
        print(f"   字符数: {chunk_char_len}")
        print(f"   Token数: {chunk_token_len}")
        print(f"   Token密度: {chunk_char_len/chunk_token_len:.2f} 字符/token")
        
        # 显示切片内容
        chunk_preview = chunk[:150].replace('\n', ' ').strip()
        if len(chunk) > 150:
            chunk_preview += "..."
        print(f"   内容: {chunk_preview}")
        
        # 显示与前一个切片的重叠情况
        if i > 0:
            prev_chunk = chunks[i-1]
            # 简单的重叠检测：检查当前块的开头是否在前一个块中出现
            overlap_chars = 0
            chunk_start = chunk[:100]  # 取当前块的前100字符
            if chunk_start in prev_chunk:
                overlap_start = prev_chunk.find(chunk_start)
                if overlap_start != -1:
                    overlap_chars = len(prev_chunk) - overlap_start
            
            if overlap_chars > 0:
                print(f"   与前块重叠: ~{overlap_chars} 字符")
    
    # 4. 统计摘要
    print_section("4. 切片统计摘要")
    
    print(f"📊 原始文本统计:")
    print(f"   总字符数: {len(text)}")
    print(f"   总Token数: {len(original_tokens)}")
    
    print(f"\n📊 切片后统计:")
    print(f"   切片数量: {len(chunks)}")
    print(f"   总字符数: {total_chunk_chars}")
    print(f"   总Token数: {total_chunk_tokens}")
    
    # 计算重复率
    char_duplication = (total_chunk_chars - len(text)) / len(text) * 100
    token_duplication = (total_chunk_tokens - len(original_tokens)) / len(original_tokens) * 100
    
    print(f"\n📊 重叠分析:")
    print(f"   字符重复率: {char_duplication:.2f}%")
    print(f"   Token重复率: {token_duplication:.2f}%")
    
    # 切片大小分布
    chunk_sizes = [len(chunker.encoder.encode(chunk, disallowed_special=())) for chunk in chunks]
    print(f"\n📊 切片大小分布:")
    print(f"   最小Token数: {min(chunk_sizes)}")
    print(f"   最大Token数: {max(chunk_sizes)}")
    print(f"   平均Token数: {sum(chunk_sizes)/len(chunk_sizes):.1f}")
    print(f"   目标Token数: {chunk_size}")
    
    # 5. 切片质量评估
    print_section("5. 切片质量评估")
    
    # 检查是否有过小的切片
    small_chunks = [i for i, size in enumerate(chunk_sizes) if size < chunk_size * 0.5]
    if small_chunks:
        print(f"⚠️  发现 {len(small_chunks)} 个较小的切片 (< 50%目标大小): {[i+1 for i in small_chunks]}")
    else:
        print("✅ 所有切片大小都在合理范围内")
    
    # 检查是否有过大的切片
    large_chunks = [i for i, size in enumerate(chunk_sizes) if size > chunk_size * 1.1]
    if large_chunks:
        print(f"⚠️  发现 {len(large_chunks)} 个较大的切片 (> 110%目标大小): {[i+1 for i in large_chunks]}")
    else:
        print("✅ 所有切片大小都在目标范围内")
    
    # 评估切片边界质量
    boundary_quality = 0
    for chunk in chunks[:-1]:  # 除了最后一个切片
        # 检查是否在句子边界结束
        if chunk.rstrip().endswith(('.', '。', '!', '！', '?', '？')):
            boundary_quality += 1
        # 检查是否在段落边界结束
        elif chunk.rstrip().endswith('\n'):
            boundary_quality += 0.8
        # 检查是否在逗号或分号结束
        elif chunk.rstrip().endswith((',', '，', ';', '；')):
            boundary_quality += 0.5
    
    boundary_score = boundary_quality / (len(chunks) - 1) * 100 if len(chunks) > 1 else 100
    print(f"📈 切片边界质量评分: {boundary_score:.1f}%")
    
    return chunks


async def main():
    """主函数"""
    print("🚀 简化文本切片演示")
    
    # 示例文本 - 混合中英文内容
    sample_text = """
    賞之味控股有限公司（「本公司」）董事會（「董事會」）謹此宣佈，本公司將於2024年9月19日（星期四）下午2時30分假座香港九龍觀塘開源道79號鱷魚恤中心6樓舉行股東週年大會（「股東大會」）。

    股東大會將審議以下事項：
    1. 省覽及採納截至2024年3月31日止年度之經審核財務報表及董事會報告書和核數師報告書；
    2. 重選退任董事；
    3. 續聘核數師及授權董事會釐定其酬金；
    4. 考慮及酌情通過授出購回股份之一般授權；
    5. 考慮及酌情通過授出發行股份之一般授權。

    所有股東均獲邀請出席股東大會。未能親身出席大會之股東可委任代表代為出席及投票。代表委任表格連同簽署授權書或其他授權文件（如有）須於股東大會舉行前不少於48小時送達本公司之股份過戶登記處卓佳證券登記有限公司，地址為香港皇后大道東183號合和中心54樓。

    Financial Summary for the Year Ended March 31, 2024:
    - Revenue: HK$1,234,567,890
    - Gross Profit: HK$234,567,890  
    - Net Profit: HK$123,456,789
    - Earnings Per Share: HK$0.12
    - Dividend Per Share: HK$0.05

    Business Outlook:
    The Company will continue to focus on core business development, enhancing competitiveness through product innovation and market expansion. Management expects to face market challenges in the coming year but remains optimistic about long-term development prospects.

    本公司將繼續專注於核心業務發展，通過產品創新和市場拓展來提升競爭力。預期未來一年將面臨市場挑戰，但管理層對長期發展前景保持樂觀態度。我們計劃在以下幾個方面加強投入：

    1. 技術研發：增加研發投入，提升產品技術含量
    2. 市場營銷：擴大品牌影響力，開拓新的市場領域
    3. 人才培養：建立完善的人才培養體系，提升團隊整體素質
    4. 供應鏈優化：優化供應鏈管理，降低運營成本
    5. 數字化轉型：推進企業數字化轉型，提高運營效率
    """
    
    try:
        # 使用不同的切片配置進行演示
        await demo_chunking(sample_text.strip(), chunk_size=256, chunk_overlap=32)
        
        print_header("演示完成", char="*")
        print("💡 提示：您可以修改 chunk_size 和 chunk_overlap 參數來測試不同的切片效果")
        print("💡 建議：對於中文文本，可以使用較大的 chunk_size (如 512-1024)")
        
    except Exception as e:
        print(f"❌ 演示過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
