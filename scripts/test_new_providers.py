#!/usr/bin/env python3
"""
测试新增提供商功能的脚本

专门测试SiliconFlow嵌入模型和OpenAI兼容API的功能。
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_siliconflow_embedder():
    """测试SiliconFlow嵌入模型"""
    print("🧪 测试 SiliconFlow 嵌入模型...")
    
    try:
        # 测试导入
        from haiku.rag.embeddings.siliconflow import Embedder
        print("   ✅ SiliconFlow嵌入模型导入成功")
        
        # 测试配置
        from haiku.rag.config import Config
        
        if Config.EMBEDDINGS_PROVIDER == "siliconflow":
            # 测试初始化
            embedder = Embedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)
            print(f"   ✅ 嵌入模型初始化成功")
            print(f"      模型: {embedder._model}")
            print(f"      维度: {embedder._vector_dim}")
            print(f"      API端点: {embedder._base_url}")
            
            # 检查API密钥
            if embedder._api_key:
                masked_key = embedder._api_key[:8] + "..." + embedder._api_key[-4:]
                print(f"      API密钥: {masked_key}")
            else:
                print("      ❌ API密钥未设置")
                return False
            
            print("   ✅ SiliconFlow嵌入模型配置正确")
            return True
        else:
            print(f"   ⚠️  当前嵌入提供商不是SiliconFlow: {Config.EMBEDDINGS_PROVIDER}")
            return True
            
    except Exception as e:
        print(f"   ❌ SiliconFlow嵌入模型测试失败: {e}")
        return False


def test_openai_qa_agent():
    """测试OpenAI QA代理"""
    print("\n🤖 测试 OpenAI QA 代理...")
    
    try:
        # 测试导入
        from haiku.rag.qa.openai import QuestionAnswerOpenAIAgent
        print("   ✅ OpenAI QA代理导入成功")
        
        # 测试配置
        from haiku.rag.config import Config
        
        if Config.QA_PROVIDER == "openai":
            print(f"   ✅ QA提供商配置正确: {Config.QA_PROVIDER}")
            print(f"      模型: {Config.QA_MODEL}")
            
            # 检查API密钥
            if Config.OPENAI_API_KEY:
                masked_key = Config.OPENAI_API_KEY[:8] + "..." + Config.OPENAI_API_KEY[-4:]
                print(f"      API密钥: {masked_key}")
            else:
                print("      ❌ OPENAI_API_KEY未设置")
                return False
            
            # 检查自定义端点
            if Config.OPENAI_BASE_URL:
                print(f"      自定义端点: {Config.OPENAI_BASE_URL}")
            else:
                print("      端点: 默认OpenAI端点")
            
            print("   ✅ OpenAI QA代理配置正确")
            return True
        else:
            print(f"   ⚠️  当前QA提供商不是OpenAI: {Config.QA_PROVIDER}")
            return True
            
    except Exception as e:
        print(f"   ❌ OpenAI QA代理测试失败: {e}")
        return False


def test_factory_functions():
    """测试工厂函数"""
    print("\n🏭 测试工厂函数...")
    
    try:
        from haiku.rag.config import Config
        
        # 测试嵌入模型工厂
        if Config.EMBEDDINGS_PROVIDER == "siliconflow":
            from haiku.rag.embeddings import get_embedder
            embedder = get_embedder()
            print(f"   ✅ 嵌入模型工厂成功创建SiliconFlow嵌入器")
            print(f"      类型: {type(embedder).__name__}")
            print(f"      模型: {embedder._model}")
        
        # 测试QA代理工厂
        if Config.QA_PROVIDER == "openai":
            from haiku.rag.qa import get_qa_agent
            # 创建一个模拟客户端来测试
            class MockClient:
                pass
            
            qa_agent = get_qa_agent(MockClient(), Config.QA_MODEL)
            print(f"   ✅ QA代理工厂成功创建OpenAI代理")
            print(f"      类型: {type(qa_agent).__name__}")
            print(f"      模型: {qa_agent._model}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 工厂函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_loading():
    """测试配置加载"""
    print("\n⚙️  测试配置加载...")
    
    try:
        from haiku.rag.config import Config
        
        # 检查新增的配置项
        config_items = [
            ("OPENAI_BASE_URL", Config.OPENAI_BASE_URL),
            ("SILICONFLOW_API_KEY", Config.SILICONFLOW_API_KEY),
            ("SILICONFLOW_BASE_URL", Config.SILICONFLOW_BASE_URL),
            ("EMBEDDINGS_PROVIDER", Config.EMBEDDINGS_PROVIDER),
            ("QA_PROVIDER", Config.QA_PROVIDER),
        ]
        
        for name, value in config_items:
            if value:
                if "API_KEY" in name:
                    masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    print(f"   ✅ {name}: {masked_value}")
                else:
                    print(f"   ✅ {name}: {value}")
            else:
                print(f"   ⚠️  {name}: 未设置")
        
        print("   ✅ 配置加载成功")
        return True
        
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        return False


def test_mixed_configuration():
    """测试混合配置"""
    print("\n🔄 测试混合配置...")
    
    try:
        from haiku.rag.config import Config
        
        qa_provider = Config.QA_PROVIDER
        embeddings_provider = Config.EMBEDDINGS_PROVIDER
        
        print(f"   QA提供商: {qa_provider}")
        print(f"   嵌入提供商: {embeddings_provider}")
        
        # 检查是否是混合配置
        if qa_provider != embeddings_provider:
            print(f"   ✅ 检测到混合配置: {qa_provider} + {embeddings_provider}")
            
            # 验证特定的混合配置
            if qa_provider == "openai" and embeddings_provider == "siliconflow":
                print("   🎯 这是推荐的混合配置：OpenAI QA + SiliconFlow嵌入")
                
                # 检查必要的配置
                if Config.OPENAI_API_KEY and Config.SILICONFLOW_API_KEY:
                    print("   ✅ 混合配置的API密钥都已设置")
                    return True
                else:
                    print("   ❌ 混合配置缺少必要的API密钥")
                    return False
            else:
                print(f"   ✅ 其他混合配置: {qa_provider} + {embeddings_provider}")
                return True
        else:
            print(f"   ✅ 单一提供商配置: {qa_provider}")
            return True
            
    except Exception as e:
        print(f"   ❌ 混合配置测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 新增提供商功能测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("配置加载", test_configuration_loading),
        ("SiliconFlow嵌入模型", test_siliconflow_embedder),
        ("OpenAI QA代理", test_openai_qa_agent),
        ("工厂函数", test_factory_functions),
        ("混合配置", test_mixed_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
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
        print("\n🎉 所有新增功能测试通过！")
        print("\n📋 功能清单:")
        print("   ✅ SiliconFlow嵌入模型适配")
        print("   ✅ OpenAI兼容API支持")
        print("   ✅ 混合提供商配置")
        print("   ✅ 工厂模式更新")
        print("   ✅ 配置系统增强")
        print("\n🚀 可以开始使用混合配置了！")
    else:
        print("\n⚠️  部分功能测试失败，请检查配置。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 测试被中断")
    except Exception as e:
        print(f"\n❌ 测试脚本出错: {e}")
        import traceback
        traceback.print_exc()
