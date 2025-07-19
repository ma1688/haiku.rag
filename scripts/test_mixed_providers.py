#!/usr/bin/env python3
"""
测试混合提供商配置的脚本

这个脚本验证混合配置（OpenAI QA + SiliconFlow 嵌入）是否正常工作。
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.embeddings import get_embedder
from haiku.rag.qa import get_qa_agent


async def test_embeddings():
    """测试嵌入模型功能"""
    print("🧪 测试 SiliconFlow 嵌入模型...")
    
    try:
        embedder = get_embedder()
        print(f"   ✅ 嵌入模型初始化成功: {embedder._model}")
        print(f"   📏 向量维度: {embedder._vector_dim}")
        
        # 测试嵌入生成
        test_text = "这是一个测试文本，用于验证嵌入模型是否正常工作。"
        embedding = await embedder.embed(test_text)
        
        print(f"   ✅ 嵌入生成成功，维度: {len(embedding)}")
        print(f"   📊 前5个值: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 嵌入模型测试失败: {e}")
        return False


async def test_qa_agent():
    """测试QA代理功能"""
    print("\n🤖 测试 OpenAI QA 代理...")
    
    try:
        # 创建临时数据库
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        async with HaikuRAG(db_path) as client:
            # 添加测试文档
            doc = await client.create_document(
                content="Python是一种高级编程语言，以其简洁和可读性而闻名。它广泛用于Web开发、数据科学、人工智能和自动化。",
                uri="python_intro.md"
            )
            print(f"   ✅ 测试文档创建成功: {doc.id}")
            
            # 测试QA代理
            qa_agent = get_qa_agent(client)
            print(f"   ✅ QA代理初始化成功: {qa_agent._model}")
            
            # 测试问答
            question = "Python是什么？"
            answer = await qa_agent.answer(question)
            
            print(f"   ✅ 问答测试成功")
            print(f"   ❓ 问题: {question}")
            print(f"   💬 回答: {answer[:100]}...")
            
        # 清理临时文件
        try:
            Path(db_path).unlink()
        except Exception:
            pass
            
        return True
        
    except Exception as e:
        print(f"   ❌ QA代理测试失败: {e}")
        return False


async def test_full_pipeline():
    """测试完整的RAG流水线"""
    print("\n🔄 测试完整RAG流水线...")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        async with HaikuRAG(db_path) as client:
            # 添加多个测试文档
            documents = [
                {
                    "content": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
                    "uri": "ml_intro.md"
                },
                {
                    "content": "深度学习是机器学习的一个子集，使用神经网络来模拟人脑的学习过程。",
                    "uri": "dl_intro.md"
                },
                {
                    "content": "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
                    "uri": "nlp_intro.md"
                }
            ]
            
            for doc_data in documents:
                doc = await client.create_document(
                    content=doc_data["content"],
                    uri=doc_data["uri"]
                )
                print(f"   📄 文档添加: {doc_data['uri']} (ID: {doc.id})")
            
            # 测试搜索功能
            search_results = await client.search("什么是机器学习？", limit=3)
            print(f"   🔍 搜索结果: {len(search_results)} 个")
            
            for i, (chunk, score) in enumerate(search_results, 1):
                print(f"      {i}. 评分: {score:.3f} | 来源: {chunk.document_uri}")
            
            # 测试问答功能
            answer = await client.ask("机器学习和深度学习有什么区别？")
            print(f"   💬 问答结果: {answer[:150]}...")
            
        # 清理临时文件
        try:
            Path(db_path).unlink()
        except Exception:
            pass
            
        return True
        
    except Exception as e:
        print(f"   ❌ 完整流水线测试失败: {e}")
        return False


def print_configuration():
    """打印当前配置"""
    print("⚙️  当前配置:")
    print(f"   QA提供商: {Config.QA_PROVIDER}")
    print(f"   QA模型: {Config.QA_MODEL}")
    print(f"   嵌入提供商: {Config.EMBEDDINGS_PROVIDER}")
    print(f"   嵌入模型: {Config.EMBEDDINGS_MODEL}")
    print(f"   向量维度: {Config.EMBEDDINGS_VECTOR_DIM}")
    print(f"   分块大小: {Config.CHUNK_SIZE}")
    print(f"   分块重叠: {Config.CHUNK_OVERLAP}")
    
    # 检查API密钥（隐藏实际值）
    openai_key = Config.OPENAI_API_KEY
    siliconflow_key = Config.SILICONFLOW_API_KEY
    
    print(f"   OpenAI API Key: {'✅ 已设置' if openai_key else '❌ 未设置'}")
    print(f"   OpenAI Base URL: {Config.OPENAI_BASE_URL or '默认'}")
    print(f"   SiliconFlow API Key: {'✅ 已设置' if siliconflow_key else '❌ 未设置'}")
    print(f"   SiliconFlow Base URL: {Config.SILICONFLOW_BASE_URL}")


async def main():
    """主函数"""
    print("🚀 混合提供商配置测试")
    print("=" * 50)
    
    # 打印配置
    print_configuration()
    print()
    
    # 检查必要的依赖
    missing_deps = []
    
    try:
        import httpx
    except ImportError:
        missing_deps.append("httpx")
    
    try:
        import openai
    except ImportError:
        missing_deps.append("openai")
    
    if missing_deps:
        print(f"❌ 缺少依赖包: {', '.join(missing_deps)}")
        print("请安装: pip install " + " ".join(missing_deps))
        return
    
    # 检查API密钥
    if not Config.OPENAI_API_KEY:
        print("❌ 缺少 OPENAI_API_KEY 环境变量")
        return
    
    if not Config.SILICONFLOW_API_KEY:
        print("❌ 缺少 SILICONFLOW_API_KEY 环境变量")
        return
    
    # 运行测试
    tests = [
        ("嵌入模型", test_embeddings),
        ("QA代理", test_qa_agent),
        ("完整流水线", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！混合配置工作正常。")
    else:
        print("\n⚠️  部分测试失败，请检查配置和网络连接。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 测试被中断")
    except Exception as e:
        print(f"\n❌ 测试脚本出错: {e}")
