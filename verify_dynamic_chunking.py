#!/usr/bin/env python
"""
验证动态切块配置是否正确应用
Verify Dynamic Chunking Configuration
"""

import sys
from pathlib import Path

# 添加源代码路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from haiku.rag.config import Config
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.engine import Store


def verify_configuration():
    """验证配置是否正确加载"""
    print("=== 验证动态切块配置 ===\n")
    
    print("当前配置:")
    print(f"  USE_FINANCIAL_CHUNKER: {Config.USE_FINANCIAL_CHUNKER}")
    print(f"  FINANCIAL_CHUNK_SIZE: {Config.FINANCIAL_CHUNK_SIZE}")
    print(f"  FINANCIAL_CHUNK_OVERLAP: {Config.FINANCIAL_CHUNK_OVERLAP}")
    print(f"  FINANCIAL_MIN_CHUNK_SIZE: {Config.FINANCIAL_MIN_CHUNK_SIZE}")
    print(f"  FINANCIAL_MAX_CHUNK_SIZE: {Config.FINANCIAL_MAX_CHUNK_SIZE}")
    print(f"  FINANCIAL_CHUNK_SIZE_VARIANCE: {Config.FINANCIAL_CHUNK_SIZE_VARIANCE}")
    print(f"  PRESERVE_TABLES: {Config.PRESERVE_TABLES}")
    print(f"  EXTRACT_METADATA: {Config.EXTRACT_METADATA}")
    print()
    
    # 验证切块器实例化
    try:
        store = Store(":memory:")
        chunk_repo = ChunkRepository(store)
        
        print("切块器实例化:")
        print(f"  类型: {type(chunk_repo.chunker).__name__}")
        
        if hasattr(chunk_repo.chunker, 'min_chunk_size'):
            print(f"  目标大小: {chunk_repo.chunker.chunk_size}")
            print(f"  最小大小: {chunk_repo.chunker.min_chunk_size}")
            print(f"  最大大小: {chunk_repo.chunker.max_chunk_size}")
            print(f"  浮动范围: {chunk_repo.chunker.chunk_size_variance}")
            print(f"  重叠大小: {chunk_repo.chunker.chunk_overlap}")
            print(f"  保护表格: {chunk_repo.chunker.preserve_tables}")
            print(f"  提取元数据: {chunk_repo.chunker.extract_metadata}")
        else:
            print("  ❌ 未使用金融切块器")
            
        print("\n✅ 配置验证成功！")
        
    except Exception as e:
        print(f"\n❌ 配置验证失败: {e}")
        return False
    
    return True


def check_parameter_validity():
    """检查参数有效性"""
    print("\n=== 参数有效性检查 ===\n")
    
    issues = []
    
    # 检查大小关系
    if Config.FINANCIAL_MIN_CHUNK_SIZE > Config.FINANCIAL_MAX_CHUNK_SIZE:
        issues.append("最小切块大小不能大于最大切块大小")
    
    if Config.FINANCIAL_CHUNK_SIZE < Config.FINANCIAL_MIN_CHUNK_SIZE:
        issues.append("目标切块大小不能小于最小切块大小")
        
    if Config.FINANCIAL_CHUNK_SIZE > Config.FINANCIAL_MAX_CHUNK_SIZE:
        issues.append("目标切块大小不能大于最大切块大小")
    
    # 检查重叠
    if Config.FINANCIAL_CHUNK_OVERLAP >= Config.FINANCIAL_MIN_CHUNK_SIZE:
        issues.append("重叠大小不应大于等于最小切块大小")
    
    # 检查浮动范围
    if Config.FINANCIAL_CHUNK_SIZE_VARIANCE > (Config.FINANCIAL_MAX_CHUNK_SIZE - Config.FINANCIAL_MIN_CHUNK_SIZE):
        issues.append("浮动范围过大，超出了最大最小值的差值")
    
    if issues:
        print("❌ 发现参数问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 所有参数都有效")
        return True


def suggest_optimizations():
    """建议优化"""
    print("\n=== 优化建议 ===\n")
    
    suggestions = []
    
    # 重叠率建议
    overlap_ratio = Config.FINANCIAL_CHUNK_OVERLAP / Config.FINANCIAL_CHUNK_SIZE
    if overlap_ratio < 0.1:
        suggestions.append(f"重叠率 ({overlap_ratio:.1%}) 较低，建议增加到10-20%")
    elif overlap_ratio > 0.3:
        suggestions.append(f"重叠率 ({overlap_ratio:.1%}) 较高，可能增加冗余")
    
    # 大小范围建议
    size_range = Config.FINANCIAL_MAX_CHUNK_SIZE - Config.FINANCIAL_MIN_CHUNK_SIZE
    if size_range < 100:
        suggestions.append(f"切块大小范围 ({size_range}) 较小，可能限制灵活性")
    elif size_range > 300:
        suggestions.append(f"切块大小范围 ({size_range}) 较大，可能影响一致性")
    
    # 浮动范围建议
    variance_ratio = Config.FINANCIAL_CHUNK_SIZE_VARIANCE / Config.FINANCIAL_CHUNK_SIZE
    if variance_ratio > 0.5:
        suggestions.append(f"浮动范围比例 ({variance_ratio:.1%}) 较大，可能影响稳定性")
    
    if suggestions:
        print("💡 优化建议:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("✅ 当前配置已经很好")


def main():
    """主函数"""
    print("🔧 Haiku RAG 动态切块配置验证工具")
    print("=" * 50)
    
    # 验证配置
    config_ok = verify_configuration()
    
    # 检查参数有效性
    params_ok = check_parameter_validity()
    
    # 建议优化
    suggest_optimizations()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 验证结果:")
    
    if config_ok and params_ok:
        print("✅ 动态切块配置验证通过！")
        print("\n🚀 下一步:")
        print("   1. 运行测试: python test_dynamic_chunking.py")
        print("   2. 处理文档: haiku-rag add /path/to/documents")
        print("   3. 开始问答: haiku-rag chat")
    else:
        print("❌ 配置存在问题，请根据上述提示进行修正。")
        print("\n📖 参考文档: docs/dynamic_chunking.md")


if __name__ == "__main__":
    main()
