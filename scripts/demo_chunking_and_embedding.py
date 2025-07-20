#!/usr/bin/env python3
"""
文本切片和向量化演示脚本

这个脚本展示了：
1. 原始文本内容
2. 文本切片后的块
3. 每个块的token数量和长度
4. 向量化后的嵌入数据
5. 切片的元数据信息
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from haiku.rag.chunker import Chunker
from haiku.rag.embeddings import get_embedder
from haiku.rag.config import Config


class ChunkingDemo:
    """文本切片和向量化演示类"""
    
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = get_embedder()
    
    def print_separator(self, title: str, char: str = "=", width: int = 80):
        """打印分隔线"""
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")
    
    def print_subsection(self, title: str, char: str = "-", width: int = 60):
        """打印子分隔线"""
        print(f"\n{char * width}")
        print(f" {title}")
        print(f"{char * width}")
    
    async def analyze_text(self, text: str, show_embeddings: bool = True) -> Dict[str, Any]:
        """分析文本的切片和向量化结果"""
        
        # 1. 显示原始文本信息
        self.print_separator("原始文本分析")
        print(f"文本长度: {len(text)} 字符")
        
        # 计算原始文本的token数量
        original_tokens = self.chunker.encoder.encode(text, disallowed_special=())
        print(f"Token数量: {len(original_tokens)} tokens")
        print(f"平均每个token字符数: {len(text) / len(original_tokens):.2f}")
        
        self.print_subsection("原始文本内容")
        # 显示文本前500字符
        preview_text = text[:500] + "..." if len(text) > 500 else text
        print(preview_text)
        
        # 2. 执行文本切片
        self.print_separator("文本切片分析")
        print(f"切片配置:")
        print(f"  - 切片大小: {self.chunker.chunk_size} tokens")
        print(f"  - 重叠大小: {self.chunker.chunk_overlap} tokens")
        
        chunks = await self.chunker.chunk(text)
        print(f"\n切片结果: 共生成 {len(chunks)} 个切片")
        
        # 分析每个切片
        chunk_analysis = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self.chunker.encoder.encode(chunk, disallowed_special=())
            chunk_info = {
                "index": i,
                "content": chunk,
                "char_length": len(chunk),
                "token_count": len(chunk_tokens),
                "token_efficiency": len(chunk) / len(chunk_tokens) if chunk_tokens else 0
            }
            chunk_analysis.append(chunk_info)
            
            self.print_subsection(f"切片 {i+1}")
            print(f"字符长度: {chunk_info['char_length']}")
            print(f"Token数量: {chunk_info['token_count']}")
            print(f"Token效率: {chunk_info['token_efficiency']:.2f} 字符/token")
            
            # 显示切片内容预览
            chunk_preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(f"内容预览: {chunk_preview}")
        
        # 3. 向量化分析
        if show_embeddings:
            self.print_separator("向量化分析")
            print(f"嵌入模型: {self.embedder._model}")
            print(f"向量维度: {self.embedder._vector_dim}")
            
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self.embedder.embed(chunk)
                    embeddings.append(embedding)
                    
                    self.print_subsection(f"切片 {i+1} 向量化结果")
                    print(f"向量维度: {len(embedding)}")
                    print(f"向量范围: [{min(embedding):.6f}, {max(embedding):.6f}]")
                    print(f"向量均值: {sum(embedding) / len(embedding):.6f}")
                    print(f"向量前10维: {embedding[:10]}")
                    
                except Exception as e:
                    print(f"切片 {i+1} 向量化失败: {e}")
                    embeddings.append(None)
        
        # 4. 统计摘要
        self.print_separator("统计摘要")
        total_chunk_chars = sum(info['char_length'] for info in chunk_analysis)
        total_chunk_tokens = sum(info['token_count'] for info in chunk_analysis)
        
        print(f"原始文本统计:")
        print(f"  - 字符数: {len(text)}")
        print(f"  - Token数: {len(original_tokens)}")
        
        print(f"\n切片后统计:")
        print(f"  - 切片数量: {len(chunks)}")
        print(f"  - 总字符数: {total_chunk_chars}")
        print(f"  - 总Token数: {total_chunk_tokens}")
        print(f"  - 字符重复率: {(total_chunk_chars - len(text)) / len(text) * 100:.2f}%")
        print(f"  - Token重复率: {(total_chunk_tokens - len(original_tokens)) / len(original_tokens) * 100:.2f}%")
        
        print(f"\n切片大小分布:")
        token_counts = [info['token_count'] for info in chunk_analysis]
        print(f"  - 最小Token数: {min(token_counts)}")
        print(f"  - 最大Token数: {max(token_counts)}")
        print(f"  - 平均Token数: {sum(token_counts) / len(token_counts):.2f}")
        
        return {
            "original_text": text,
            "original_stats": {
                "char_length": len(text),
                "token_count": len(original_tokens)
            },
            "chunks": chunk_analysis,
            "embeddings": embeddings if show_embeddings else None,
            "summary": {
                "chunk_count": len(chunks),
                "total_chunk_chars": total_chunk_chars,
                "total_chunk_tokens": total_chunk_tokens,
                "char_duplication_rate": (total_chunk_chars - len(text)) / len(text),
                "token_duplication_rate": (total_chunk_tokens - len(original_tokens)) / len(original_tokens)
            }
        }


async def main():
    """主函数"""
    print("文本切片和向量化演示脚本")
    print(f"当前配置:")
    print(f"  - 嵌入提供商: {Config.EMBEDDINGS_PROVIDER}")
    print(f"  - 嵌入模型: {Config.EMBEDDINGS_MODEL}")
    print(f"  - 向量维度: {Config.EMBEDDINGS_VECTOR_DIM}")
    
    # 示例文本 - 中文金融文档
    sample_text = """
    賞之味控股有限公司（「本公司」）董事會（「董事會」）謹此宣佈，本公司將於2024年9月19日（星期四）
    下午2時30分假座香港九龍觀塘開源道79號鱷魚恤中心6樓舉行股東週年大會（「股東大會」）。
    
    股東大會將審議以下事項：
    1. 省覽及採納截至2024年3月31日止年度之經審核財務報表及董事會報告書和核數師報告書；
    2. 重選退任董事；
    3. 續聘核數師及授權董事會釐定其酬金；
    4. 考慮及酌情通過授出購回股份之一般授權；
    5. 考慮及酌情通過授出發行股份之一般授權。
    
    所有股東均獲邀請出席股東大會。未能親身出席大會之股東可委任代表代為出席及投票。
    代表委任表格連同簽署授權書或其他授權文件（如有）須於股東大會舉行前不少於48小時
    送達本公司之股份過戶登記處卓佳證券登記有限公司，地址為香港皇后大道東183號
    合和中心54樓。
    
    本公司截至2024年3月31日止年度之年報將於適當時候寄發予股東。
    
    財務摘要：
    - 營業收入：港幣1,234,567,890元
    - 毛利潤：港幣234,567,890元  
    - 淨利潤：港幣123,456,789元
    - 每股盈利：港幣0.12元
    - 股息：每股港幣0.05元
    
    業務展望：
    本公司將繼續專注於核心業務發展，通過產品創新和市場拓展來提升競爭力。
    預期未來一年將面臨市場挑戰，但管理層對長期發展前景保持樂觀態度。
    """
    
    demo = ChunkingDemo()
    
    # 分析示例文本
    try:
        result = await demo.analyze_text(sample_text.strip(), show_embeddings=True)
        
        # 可选：保存结果到JSON文件
        output_file = "chunking_demo_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 移除嵌入向量以减少文件大小
            result_copy = result.copy()
            if result_copy.get('embeddings'):
                result_copy['embeddings'] = [
                    f"向量维度: {len(emb)}, 范围: [{min(emb):.6f}, {max(emb):.6f}]" 
                    if emb else None 
                    for emb in result_copy['embeddings']
                ]
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
