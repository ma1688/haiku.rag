#!/usr/bin/env python
"""
演示金融领域专用提示词系统的使用
Demo: Financial Domain-Specific Prompting System
"""

import asyncio
import os
from pathlib import Path
import sys

# 添加源代码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.financial_qa import (
    FinancialQuestionAnswerAgent,
    FinancialQuestionAnswerOpenAIAgent,
    FinancialQuestionAnswerOllamaAgent
)
from haiku.rag.qa.financial_prompts import get_intent_prompt


async def demo_intent_detection():
    """演示查询意图识别"""
    print("=== 查询意图识别演示 ===\n")
    
    test_queries = [
        "腾讯2023年的营收和利润是多少？",
        "分析长江和记收购英国电信的交易结构",
        "这笔交易是否构成关连交易？需要股东批准吗？",
        "比较阿里巴巴和京东的市盈率",
        "What is the EBITDA margin?",
        "Tell me about the acquisition terms and conditions"
    ]
    
    for query in test_queries:
        intent = get_intent_prompt(query)
        intent_type = "通用查询"
        
        if "FINANCIAL_DATA" in intent:
            intent_type = "财务数据提取"
        elif "TRANSACTION_ANALYSIS" in intent:
            intent_type = "交易分析"
        elif "COMPLIANCE_CHECK" in intent:
            intent_type = "合规检查"
        elif "COMPARATIVE_ANALYSIS" in intent:
            intent_type = "比较分析"
        
        print(f"查询：{query}")
        print(f"意图：{intent_type}")
        print("-" * 50)


async def demo_with_sample_data():
    """使用示例数据演示金融问答"""
    print("\n=== 金融问答系统演示 ===\n")
    
    # 创建内存数据库
    async with HaikuRAG(":memory:") as client:
        # 添加示例公告
        sample_announcements = [
            """
香港交易所上市公司公告
股份代號：0700
公司名稱：騰訊控股有限公司

2023年度盈利公告

財務摘要：
- 總收入：人民幣6,090億元，同比增長10%
- 毛利：人民幣2,921億元，毛利率48%
- 年度盈利：人民幣1,882億元，同比增長36%
- 每股基本盈利：人民幣19.92元
- EBITDA：人民幣2,511億元，EBITDA利潤率41%

董事會建議派發末期股息每股港幣3.40元。
""",
            """
香港交易所上市公司公告
股份代號：0001
公司名稱：長江和記實業有限公司

主要及關連交易
收購英國電信業務

交易詳情：
- 收購目標：UK Telecom Limited 100%股權
- 交易代價：港幣150億元
- 支付方式：現金，分三期支付
  - 首期40%（港幣60億元）：簽約時
  - 第二期30%（港幣45億元）：監管批准後
  - 尾款30%（港幣45億元）：交割完成時

關連交易性質：
賣方為本公司主要股東的聯繫人，構成關連交易。
根據上市規則第14A章，需要獨立股東批准。

財務影響：
- 預期年收入貢獻：港幣20億元
- 預期EBITDA貢獻：港幣5億元
- 預期投資回報率：15-18%
"""
        ]
        
        # 添加文档
        for i, content in enumerate(sample_announcements, 1):
            await client.create_document(
                content=content,
                uri=f"announcement_{i}.txt",
                metadata={"source": f"HKEX Announcement {i}"}
            )
        
        # 创建金融问答代理
        agent = FinancialQuestionAnswerAgent(client)
        
        # 测试不同类型的查询
        test_queries = [
            "腾讯2023年的营收是多少？",
            "长江和记收购的交易金额和支付安排是什么？",
            "这笔收购是否需要股东批准？",
            "腾讯的EBITDA利润率是多少？",
            "收购UK Telecom的预期回报率？"
        ]
        
        for query in test_queries:
            print(f"\n问题：{query}")
            print("-" * 80)
            
            answer = await agent.answer(query)
            print(answer)
            print("=" * 80)


async def demo_comparison():
    """演示通用提示词 vs 金融提示词的对比"""
    print("\n=== 通用 vs 金融提示词对比 ===\n")
    
    # 示例查询
    query = "分析这笔交易的财务影响和风险"
    
    print("查询：", query)
    print("\n通用提示词：")
    print("-" * 50)
    print("你是一位专业的文档分析助手...")
    print("【工作流程】")
    print("1. 使用 search_documents 工具检索相关文档")
    print("2. 仔细阅读每个检索结果")
    print("3. 从文档中寻找答案")
    
    print("\n金融专用提示词：")
    print("-" * 50)
    print("你是一位专业的金融文档分析专家，专门处理港交所上市公司公告...")
    print("【专业能力】")
    print("1. 理解金融术语和概念")
    print("2. 识别和提取关键财务数据")
    print("3. 分析交易结构和条款")
    print("4. 评估监管合规要求")
    print("\n【影响分析要点】")
    print("1. 对公司业务的影响")
    print("2. 对财务状况的影响")
    print("3. 对股权结构的影响")
    print("4. 潜在的协同效应")


async def demo_structured_output():
    """演示结构化输出格式"""
    print("\n=== 结构化输出格式演示 ===\n")
    
    # 财务数据格式
    print("1. 财务数据格式：")
    print("-" * 50)
    print("""
【騰訊控股有限公司】盈利公告

📊 关键财务数据：
- 总收入：人民幣 6,090 億元（+10%）
- 年度盈利：人民幣 1,882 億元（+36%）
- 毛利率：48%
- EBITDA利润率：41%

📅 重要日期：
- 公告日期：2024年3月20日
- 派息日期：2024年5月15日

📝 数据说明：
- 所有数据经审计
- 同比增长基于2022年数据

🔗 信息来源：
基于文档ID: 1234, 2345
""")
    
    # 交易摘要格式
    print("\n2. 交易摘要格式：")
    print("-" * 50)
    print("""
【交易概要】收购UK Telecom Limited

🏢 交易各方：
- 买方：長江和記實業有限公司
- 卖方：关连人士（主要股东联系人）
- 目标：UK Telecom Limited

💰 交易要素：
- 交易金额：港幣 150 億元
- 股权比例：100%
- 支付方式：现金分期

📋 关键条款：
- 分三期支付（40%-30%-30%）
- 需要独立股东批准
- 需要监管批准

⚠️ 风险提示：
- 关连交易合规风险
- 整合执行风险
- 监管审批不确定性

✅ 后续步骤：
1. 召开独立董事会议
2. 聘请独立财务顾问
3. 召开股东大会
4. 申请监管批准

🔗 信息来源：
基于2份相关文档
""")


async def demo_error_handling():
    """演示错误处理"""
    print("\n=== 错误处理演示 ===\n")
    
    error_scenarios = [
        {
            "scenario": "未找到相关信息",
            "message": "根据检索到的文档，未找到关于小米汽车IPO的相关信息。建议：\n1. 尝试使用不同的关键词\n2. 检查公司名称或股票代码是否正确\n3. 确认时间范围是否合适"
        },
        {
            "scenario": "信息不完整",
            "message": "找到部分相关信息，但以下数据缺失：\n- 交易金额\n- 完成日期\n\n已获得信息：\n- 交易类型：收购\n- 目标公司：ABC Limited"
        },
        {
            "scenario": "查询不明确",
            "message": "查询需求不够明确。请说明：\n1. 具体的公司名称或股票代码\n2. 需要查询的信息类型\n3. 相关的时间范围"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"场景：{scenario['scenario']}")
        print("-" * 50)
        print(scenario['message'])
        print()


async def demo_configuration():
    """演示配置和使用方法"""
    print("\n=== 配置和使用方法 ===\n")
    
    print("1. 环境变量配置：")
    print("-" * 50)
    print("""
# 启用金融问答系统
export USE_FINANCIAL_QA=true

# 设置模型（可选）
export FINANCIAL_QA_MODEL=gpt-4

# 同时启用金融切块器（推荐）
export USE_FINANCIAL_CHUNKER=true
""")
    
    print("\n2. 程序内使用：")
    print("-" * 50)
    print("""
from haiku.rag.client import HaikuRAG
from haiku.rag.qa.financial_qa import FinancialQuestionAnswerOpenAIAgent

# 创建客户端
async with HaikuRAG() as client:
    # 创建金融问答代理
    agent = FinancialQuestionAnswerOpenAIAgent(client, model="gpt-4")
    
    # 提问
    answer = await agent.answer("分析最新的收购公告")
    print(answer)
""")
    
    print("\n3. CLI 使用（需要集成）：")
    print("-" * 50)
    print("""
# 使用金融模式
haiku-rag chat --financial

# 或通过环境变量
export USE_FINANCIAL_QA=true
haiku-rag chat
""")


async def main():
    """运行所有演示"""
    print("港交所公告金融提示词系统演示")
    print("=" * 80)
    
    # 运行各个演示
    await demo_intent_detection()
    await demo_with_sample_data()
    await demo_comparison()
    await demo_structured_output()
    await demo_error_handling()
    await demo_configuration()
    
    print("\n演示完成！")
    print("\n主要优势：")
    print("1. 自动识别查询意图，选择合适的提示词")
    print("2. 理解金融术语，提取结构化数据")
    print("3. 提供专业的分析框架")
    print("4. 输出格式化的、易读的结果")
    print("5. 完善的错误处理和用户引导")


if __name__ == "__main__":
    asyncio.run(main())