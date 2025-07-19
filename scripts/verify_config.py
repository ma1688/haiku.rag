#!/usr/bin/env python3
"""
快速验证配置脚本

验证您的混合提供商配置是否正确设置。
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.config import Config


def check_dependencies():
    """检查必要的依赖包"""
    print("🔍 检查依赖包...")
    
    deps = {
        "httpx": "SiliconFlow 嵌入模型",
        "openai": "OpenAI QA 和嵌入模型",
        "anthropic": "Anthropic QA 模型",
        "voyageai": "VoyageAI 嵌入模型"
    }
    
    available = {}
    for dep, description in deps.items():
        try:
            __import__(dep)
            available[dep] = True
            print(f"   ✅ {dep}: 已安装 ({description})")
        except ImportError:
            available[dep] = False
            print(f"   ❌ {dep}: 未安装 ({description})")
    
    return available


def check_configuration():
    """检查配置设置"""
    print("\n⚙️  检查配置...")
    
    # 基本配置
    print(f"   QA提供商: {Config.QA_PROVIDER}")
    print(f"   QA模型: {Config.QA_MODEL}")
    print(f"   嵌入提供商: {Config.EMBEDDINGS_PROVIDER}")
    print(f"   嵌入模型: {Config.EMBEDDINGS_MODEL}")
    print(f"   向量维度: {Config.EMBEDDINGS_VECTOR_DIM}")
    
    # API密钥检查
    api_keys = {
        "OPENAI_API_KEY": Config.OPENAI_API_KEY,
        "SILICONFLOW_API_KEY": Config.SILICONFLOW_API_KEY,
        "ANTHROPIC_API_KEY": Config.ANTHROPIC_API_KEY,
        "VOYAGE_API_KEY": Config.VOYAGE_API_KEY
    }
    
    print("\n🔑 API密钥状态:")
    for key_name, key_value in api_keys.items():
        if key_value:
            masked_key = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
            print(f"   ✅ {key_name}: {masked_key}")
        else:
            print(f"   ❌ {key_name}: 未设置")
    
    # URL配置
    print("\n🌐 API端点:")
    if Config.OPENAI_BASE_URL:
        print(f"   OpenAI Base URL: {Config.OPENAI_BASE_URL}")
    else:
        print(f"   OpenAI Base URL: 默认 (https://api.openai.com/v1)")
    
    print(f"   SiliconFlow Base URL: {Config.SILICONFLOW_BASE_URL}")
    print(f"   Ollama Base URL: {Config.OLLAMA_BASE_URL}")


def validate_provider_combination():
    """验证提供商组合的有效性"""
    print("\n✅ 验证提供商组合...")
    
    qa_provider = Config.QA_PROVIDER
    embeddings_provider = Config.EMBEDDINGS_PROVIDER
    
    # 检查QA提供商配置
    qa_valid = True
    if qa_provider == "openai":
        if not Config.OPENAI_API_KEY:
            print(f"   ❌ OpenAI QA需要设置 OPENAI_API_KEY")
            qa_valid = False
        else:
            print(f"   ✅ OpenAI QA配置正确")
    elif qa_provider == "anthropic":
        if not Config.ANTHROPIC_API_KEY:
            print(f"   ❌ Anthropic QA需要设置 ANTHROPIC_API_KEY")
            qa_valid = False
        else:
            print(f"   ✅ Anthropic QA配置正确")
    elif qa_provider == "ollama":
        print(f"   ✅ Ollama QA配置正确 (本地模型)")
    else:
        print(f"   ❌ 不支持的QA提供商: {qa_provider}")
        qa_valid = False
    
    # 检查嵌入提供商配置
    embeddings_valid = True
    if embeddings_provider == "siliconflow":
        if not Config.SILICONFLOW_API_KEY:
            print(f"   ❌ SiliconFlow 嵌入需要设置 SILICONFLOW_API_KEY")
            embeddings_valid = False
        else:
            print(f"   ✅ SiliconFlow 嵌入配置正确")
    elif embeddings_provider == "openai":
        if not Config.OPENAI_API_KEY:
            print(f"   ❌ OpenAI 嵌入需要设置 OPENAI_API_KEY")
            embeddings_valid = False
        else:
            print(f"   ✅ OpenAI 嵌入配置正确")
    elif embeddings_provider == "voyageai":
        if not Config.VOYAGE_API_KEY:
            print(f"   ❌ VoyageAI 嵌入需要设置 VOYAGE_API_KEY")
            embeddings_valid = False
        else:
            print(f"   ✅ VoyageAI 嵌入配置正确")
    elif embeddings_provider == "ollama":
        print(f"   ✅ Ollama 嵌入配置正确 (本地模型)")
    else:
        print(f"   ❌ 不支持的嵌入提供商: {embeddings_provider}")
        embeddings_valid = False
    
    return qa_valid and embeddings_valid


def suggest_improvements():
    """建议配置改进"""
    print("\n💡 配置建议:")
    
    # 分块大小建议
    if Config.CHUNK_SIZE < 512:
        print(f"   💭 当前分块大小 ({Config.CHUNK_SIZE}) 较小，可能影响上下文质量")
        print(f"      建议设置为 1024 或更大")
    elif Config.CHUNK_SIZE > 2048:
        print(f"   💭 当前分块大小 ({Config.CHUNK_SIZE}) 较大，可能影响检索精度")
        print(f"      建议设置为 1024-2048 之间")
    else:
        print(f"   ✅ 分块大小 ({Config.CHUNK_SIZE}) 设置合理")
    
    # 重叠建议
    overlap_ratio = Config.CHUNK_OVERLAP / Config.CHUNK_SIZE
    if overlap_ratio < 0.1:
        print(f"   💭 分块重叠比例 ({overlap_ratio:.1%}) 较小，可能丢失上下文")
        print(f"      建议设置重叠为分块大小的 10-20%")
    elif overlap_ratio > 0.3:
        print(f"   💭 分块重叠比例 ({overlap_ratio:.1%}) 较大，可能增加冗余")
        print(f"      建议设置重叠为分块大小的 10-20%")
    else:
        print(f"   ✅ 分块重叠比例 ({overlap_ratio:.1%}) 设置合理")
    
    # 向量维度建议
    model_dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "voyage-3.5": 1024,
        "Qwen/Qwen3-Embedding-8B": 4096,
        "mxbai-embed-large": 1024
    }
    
    expected_dim = model_dims.get(Config.EMBEDDINGS_MODEL)
    if expected_dim and expected_dim != Config.EMBEDDINGS_VECTOR_DIM:
        print(f"   ⚠️  向量维度可能不匹配:")
        print(f"      模型 {Config.EMBEDDINGS_MODEL} 预期维度: {expected_dim}")
        print(f"      当前配置维度: {Config.EMBEDDINGS_VECTOR_DIM}")


def main():
    """主函数"""
    print("🔧 Haiku RAG 配置验证工具")
    print("=" * 50)
    
    # 检查依赖
    deps = check_dependencies()
    
    # 检查配置
    check_configuration()
    
    # 验证提供商组合
    config_valid = validate_provider_combination()
    
    # 建议改进
    suggest_improvements()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 验证结果:")
    
    if config_valid:
        print("✅ 配置验证通过！可以开始使用 Haiku RAG。")
        print("\n🚀 下一步:")
        print("   1. 运行测试: python scripts/test_mixed_providers.py")
        print("   2. 添加文档: haiku-rag add /path/to/documents")
        print("   3. 开始问答: haiku-rag chat")
    else:
        print("❌ 配置存在问题，请根据上述提示进行修正。")
        print("\n📖 参考文档: docs/configuration.md")
    
    # 检查必要依赖
    qa_provider = Config.QA_PROVIDER
    embeddings_provider = Config.EMBEDDINGS_PROVIDER
    
    missing_deps = []
    if qa_provider == "openai" and not deps.get("openai"):
        missing_deps.append("openai")
    if qa_provider == "anthropic" and not deps.get("anthropic"):
        missing_deps.append("anthropic")
    if embeddings_provider == "siliconflow" and not deps.get("httpx"):
        missing_deps.append("httpx")
    if embeddings_provider == "voyageai" and not deps.get("voyageai"):
        missing_deps.append("voyageai")
    
    if missing_deps:
        print(f"\n📦 需要安装的依赖包:")
        print(f"   pip install {' '.join(missing_deps)}")


if __name__ == "__main__":
    main()
