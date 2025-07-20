#!/usr/bin/env python3
"""
向量化演示脚本

这个脚本专门展示文本向量化功能：
1. 文本向量化过程
2. 向量数据分析
3. 向量相似度计算
4. 向量可视化（可选）
"""

import asyncio
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from haiku.rag.embeddings import get_embedder
from haiku.rag.config import Config


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


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    if len(vec1) != len(vec2):
        raise ValueError("向量维度不匹配")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def analyze_vector(vector: List[float], name: str = "向量") -> Dict[str, Any]:
    """分析向量的统计特性"""
    if not vector:
        return {}
    
    return {
        "name": name,
        "dimension": len(vector),
        "min_value": min(vector),
        "max_value": max(vector),
        "mean": sum(vector) / len(vector),
        "magnitude": math.sqrt(sum(x * x for x in vector)),
        "non_zero_count": sum(1 for x in vector if abs(x) > 1e-10),
        "sparsity": 1 - (sum(1 for x in vector if abs(x) > 1e-10) / len(vector))
    }


async def demo_embeddings():
    """演示向量化功能"""
    
    print_header("文本向量化演示")
    
    # 检查嵌入配置
    print_section("1. 嵌入模型配置")
    print(f"提供商: {Config.EMBEDDINGS_PROVIDER}")
    print(f"模型: {Config.EMBEDDINGS_MODEL}")
    print(f"向量维度: {Config.EMBEDDINGS_VECTOR_DIM}")
    
    try:
        embedder = get_embedder()
        print(f"✅ 嵌入模型加载成功")
    except Exception as e:
        print(f"❌ 嵌入模型加载失败: {e}")
        print("💡 请检查环境变量配置或安装相应的依赖包")
        return
    
    # 测试文本
    test_texts = [
        "賞之味控股有限公司將舉行股東週年大會",
        "The company will hold an annual general meeting",
        "公司財務表現良好，營收增長顯著",
        "Financial performance is strong with significant revenue growth",
        "董事會建議派發股息每股0.05港元",
        "The board recommends a dividend of HK$0.05 per share"
    ]
    
    print_section("2. 文本向量化過程")
    
    embeddings = []
    for i, text in enumerate(test_texts):
        print(f"\n📝 文本 {i+1}: {text}")
        
        try:
            embedding = await embedder.embed(text)
            embeddings.append(embedding)
            
            # 分析向量
            stats = analyze_vector(embedding, f"文本{i+1}")
            print(f"   ✅ 向量化成功")
            print(f"   📊 維度: {stats['dimension']}")
            print(f"   📊 範圍: [{stats['min_value']:.6f}, {stats['max_value']:.6f}]")
            print(f"   📊 均值: {stats['mean']:.6f}")
            print(f"   📊 模長: {stats['magnitude']:.6f}")
            print(f"   📊 稀疏度: {stats['sparsity']:.2%}")
            
            # 显示向量的前10个和后10个维度
            print(f"   🔢 前10維: {embedding[:10]}")
            if len(embedding) > 10:
                print(f"   🔢 後10維: {embedding[-10:]}")
                
        except Exception as e:
            print(f"   ❌ 向量化失败: {e}")
            embeddings.append(None)
    
    # 计算相似度矩阵
    print_section("3. 向量相似度分析")
    
    valid_embeddings = [(i, emb) for i, emb in enumerate(embeddings) if emb is not None]
    
    if len(valid_embeddings) >= 2:
        print("📊 余弦相似度矩阵:")
        print("   ", end="")
        for i, _ in valid_embeddings:
            print(f"文本{i+1:2d}", end="  ")
        print()
        
        for i, (idx1, emb1) in enumerate(valid_embeddings):
            print(f"文本{idx1+1:2d}", end=" ")
            for j, (idx2, emb2) in enumerate(valid_embeddings):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = cosine_similarity(emb1, emb2)
                print(f"{similarity:6.3f}", end="  ")
            print()
        
        # 找出最相似和最不相似的文本对
        max_similarity = -1
        min_similarity = 2
        max_pair = None
        min_pair = None
        
        for i, (idx1, emb1) in enumerate(valid_embeddings):
            for j, (idx2, emb2) in enumerate(valid_embeddings):
                if i < j:  # 避免重复计算
                    similarity = cosine_similarity(emb1, emb2)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_pair = (idx1, idx2)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        min_pair = (idx1, idx2)
        
        if max_pair:
            print(f"\n🔥 最相似的文本對:")
            print(f"   文本{max_pair[0]+1}: {test_texts[max_pair[0]]}")
            print(f"   文本{max_pair[1]+1}: {test_texts[max_pair[1]]}")
            print(f"   相似度: {max_similarity:.4f}")
        
        if min_pair:
            print(f"\n❄️  最不相似的文本對:")
            print(f"   文本{min_pair[0]+1}: {test_texts[min_pair[0]]}")
            print(f"   文本{min_pair[1]+1}: {test_texts[min_pair[1]]}")
            print(f"   相似度: {min_similarity:.4f}")
    
    # 向量分布分析
    print_section("4. 向量分布分析")
    
    if valid_embeddings:
        all_values = []
        for _, emb in valid_embeddings:
            all_values.extend(emb)
        
        print(f"📊 所有向量值統計:")
        print(f"   總數據點: {len(all_values)}")
        print(f"   最小值: {min(all_values):.6f}")
        print(f"   最大值: {max(all_values):.6f}")
        print(f"   均值: {sum(all_values)/len(all_values):.6f}")
        
        # 计算分布
        positive_count = sum(1 for x in all_values if x > 0)
        negative_count = sum(1 for x in all_values if x < 0)
        zero_count = len(all_values) - positive_count - negative_count
        
        print(f"   正值比例: {positive_count/len(all_values):.2%}")
        print(f"   負值比例: {negative_count/len(all_values):.2%}")
        print(f"   零值比例: {zero_count/len(all_values):.2%}")
        
        # 简单的分布直方图
        print(f"\n📈 值分布直方图 (10個區間):")
        min_val, max_val = min(all_values), max(all_values)
        bin_width = (max_val - min_val) / 10
        bins = [0] * 10
        
        for val in all_values:
            bin_idx = min(int((val - min_val) / bin_width), 9)
            bins[bin_idx] += 1
        
        max_count = max(bins)
        for i, count in enumerate(bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bar_length = int(count / max_count * 40) if max_count > 0 else 0
            bar = "█" * bar_length
            print(f"   [{bin_start:7.3f}, {bin_end:7.3f}): {bar} ({count})")
    
    print_section("5. 向量化質量評估")
    
    if len(valid_embeddings) >= 2:
        # 评估向量的区分度
        similarities = []
        for i, (idx1, emb1) in enumerate(valid_embeddings):
            for j, (idx2, emb2) in enumerate(valid_embeddings):
                if i < j:
                    similarities.append(cosine_similarity(emb1, emb2))
        
        avg_similarity = sum(similarities) / len(similarities)
        similarity_variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
        
        print(f"📊 向量區分度分析:")
        print(f"   平均相似度: {avg_similarity:.4f}")
        print(f"   相似度方差: {similarity_variance:.6f}")
        print(f"   相似度標準差: {math.sqrt(similarity_variance):.4f}")
        
        # 评估标准
        if avg_similarity < 0.3:
            print("   ✅ 向量區分度良好 (平均相似度 < 0.3)")
        elif avg_similarity < 0.7:
            print("   ⚠️  向量區分度中等 (0.3 ≤ 平均相似度 < 0.7)")
        else:
            print("   ❌ 向量區分度較差 (平均相似度 ≥ 0.7)")
        
        if math.sqrt(similarity_variance) > 0.1:
            print("   ✅ 向量差異性良好 (標準差 > 0.1)")
        else:
            print("   ⚠️  向量差異性較小 (標準差 ≤ 0.1)")


async def main():
    """主函数"""
    print("🚀 文本向量化演示")
    
    try:
        await demo_embeddings()
        
        print_header("演示完成", char="*")
        print("💡 提示：不同的嵌入模型會產生不同的向量表示")
        print("💡 建議：選擇適合您語言和領域的嵌入模型以獲得最佳效果")
        
    except Exception as e:
        print(f"❌ 演示過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
