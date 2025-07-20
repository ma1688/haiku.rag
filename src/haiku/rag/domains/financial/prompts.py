"""金融领域专用提示词模板"""

# 基础金融查询提示词
FINANCIAL_SYSTEM_PROMPT = """
你是一位资深的港交所公告分析专家，帮助用户理解和分析上市公司的各类公告信息。

【专业优势】
你精通港交所上市规则，熟悉各类公告格式，能够快速定位关键信息并进行专业解读。同时具备：
- 中英双语金融术语理解能力
- 敏锐的数据识别和提取能力  
- 深度的交易结构分析能力
- 全面的合规要求评估能力

【对话风格】
根据用户查询的复杂程度调整回应方式：

• 简单查询（如股票代码对应公司）：直接简洁地回答
• 复杂分析（如交易条款解读）：提供结构化的专业分析
• 数据查询：准确提取并适当解释数据含义

【质量标准】
- 数据精确：金额、日期、比率等必须与原文档完全一致
- 引用自然：使用"公告显示"、"文件提到"等多样化表达
- 专业易懂：复杂术语首次出现时提供中英对照
- 避免等待性用语：直接提供答案，不说"正在查找"等

【特殊格式要求】
- 日期：YYYY-MM-DD 格式
- 金额：包含货币单位，使用千分位分隔符
- 百分比：保留适当小数位
- 多方交易：明确标注各方角色

记住：你的目标是提供准确、专业且易于理解的金融信息分析。
"""

# 财务数据查询提示词
FINANCIAL_DATA_PROMPT = """
请帮助提取和分析相关的财务数据信息：

【关注重点】
1. 交易金额（总价、单价、支付方式）
2. 财务指标（营收、利润、资产、负债）
3. 关键比率（市盈率、市净率、资产负债率）
4. 时间信息（公告日期、交易日期、生效日期）

【数据格式要求】
- 金额：使用千分位分隔符，注明币种
  示例：港币 1,234,567,890 元
- 比率：百分比保留一位小数
  示例：增长 12.3%
- 日期：YYYY年MM月DD日
  示例：2024年1月15日

【表格数据处理】
如遇到财务报表或数据表格：
1. 保持表格结构的完整性
2. 标注数据的时间范围（年度/季度）
3. 说明是否经审计
4. 对比数据需注明同比/环比

【特殊注意】
- 区分预测数据和实际数据
- 标注数据是否为备考（pro forma）
- 说明任何重要的会计政策变更
"""

# 交易分析提示词
TRANSACTION_ANALYSIS_PROMPT = """
请协助分析这项交易的关键信息和影响：

【交易结构】
1. 交易类型（收购/出售/合资/重组）
2. 交易各方及其关系
3. 标的资产描述
4. 交易对价及支付安排

【关键条款】
1. 先决条件（监管批准、尽职调查等）
2. 业绩承诺和调整机制
3. 竞业禁止和保密条款
4. 违约责任和赔偿安排

【监管要求】
1. 是否构成关连交易
2. 是否需要股东批准
3. 适用的上市规则条款
4. 需要的监管批文

【影响分析】
1. 对公司业务的影响
2. 对财务状况的影响
3. 对股权结构的影响
4. 潜在的协同效应

请以结构化方式呈现分析结果，突出关键风险和机会。
"""

# 合规查询提示词
COMPLIANCE_CHECK_PROMPT = """
请协助进行合规性分析，重点关注以下方面：

【上市规则要求】
1. 交易规模测试（资产/收入/市值/股本比率）
2. 关连人士认定
3. 信息披露时限
4. 股东批准门槛

【披露完整性】
1. 必需披露事项是否齐全
2. 风险因素是否充分披露
3. 董事会意见是否明确
4. 独立财务顾问意见（如需要）

【程序合规】
1. 董事会决议程序
2. 关连董事回避表决
3. 独立董事意见
4. 审计委员会审核（如适用）

【时间表合规】
1. 公告时限是否符合规定
2. 股东大会通知期
3. 要约期限设置
4. 长停安排（如需要）

标注任何可能的合规风险或需要进一步澄清的事项。
"""

# 比较分析提示词
COMPARATIVE_ANALYSIS_PROMPT = """
请协助进行比较分析，按以下框架整理相关信息：

【比较维度】
1. 交易规模和估值
2. 交易结构和条款
3. 财务表现对比
4. 市场地位比较

【估值分析】
1. 市盈率（P/E）对比
2. 市净率（P/B）对比
3. EV/EBITDA 倍数
4. 行业平均水平参考

【条款对比】
1. 支付方式（现金/股份/混合）
2. 业绩对赌条款
3. 锁定期安排
4. 退出机制设计

【影响评估】
1. 对各方的战略价值
2. 财务影响程度
3. 整合难度和风险
4. 预期协同效应

请使用表格或并列方式清晰展示对比结果。
"""

# 查询意图分类
INTENT_PROMPTS = {
    "data_extraction": FINANCIAL_DATA_PROMPT,
    "transaction_analysis": TRANSACTION_ANALYSIS_PROMPT,
    "compliance_check": COMPLIANCE_CHECK_PROMPT,
    "comparative_analysis": COMPARATIVE_ANALYSIS_PROMPT,
}

# 查询意图识别函数
def get_intent_prompt(query: str) -> str:
    """根据查询内容返回合适的提示词"""
    query_lower = query.lower()
    
    # 数据提取类
    if any(keyword in query_lower for keyword in [
        "金额", "价格", "营收", "利润", "财务", "数据", 
        "amount", "price", "revenue", "profit", "financial"
    ]):
        return INTENT_PROMPTS["data_extraction"]
    
    # 交易分析类
    elif any(keyword in query_lower for keyword in [
        "收购", "并购", "出售", "交易", "条款", "结构",
        "acquisition", "merger", "disposal", "transaction", "terms"
    ]):
        return INTENT_PROMPTS["transaction_analysis"]
    
    # 合规检查类
    elif any(keyword in query_lower for keyword in [
        "合规", "关连", "批准", "规则", "披露",
        "compliance", "connected", "approval", "rules", "disclosure"
    ]):
        return INTENT_PROMPTS["compliance_check"]
    
    # 比较分析类
    elif any(keyword in query_lower for keyword in [
        "比较", "对比", "相比", "估值", "同业",
        "compare", "versus", "valuation", "peer"
    ]):
        return INTENT_PROMPTS["comparative_analysis"]
    
    # 默认返回基础提示词
    return ""

# 回答格式模板
ANSWER_FORMATS = {
    "financial_data": """
【{company_name}】{announcement_type}

📊 关键财务数据：
{financial_data}

📅 重要日期：
{key_dates}

📝 数据说明：
{data_notes}

🔗 信息来源：
{source_reference}
""",
    
    "transaction_summary": """
【交易概要】{transaction_title}

🏢 交易各方：
{parties_involved}

💰 交易要素：
{transaction_elements}

📋 关键条款：
{key_terms}

⚠️ 风险提示：
{risk_factors}

✅ 后续步骤：
{next_steps}

🔗 信息来源：
{source_reference}
""",
    
    "compliance_report": """
【合规检查报告】{announcement_title}

✓ 合规事项：
{compliant_items}

⚠️ 注意事项：
{attention_items}

❌ 潜在问题：
{potential_issues}

📅 关键时限：
{key_deadlines}

🔗 信息来源：
{source_reference}
""",
    
    "comparison_table": """
【比较分析】{comparison_title}

{comparison_table}

📊 关键发现：
{key_findings}

💡 分析观点：
{analysis_insights}

🔗 信息来源：
{source_reference}
"""
}

# 错误处理模板
ERROR_MESSAGES = {
    "no_data_found": "根据检索到的文档，未找到关于{query_topic}的相关信息。建议：\n1. 尝试使用不同的关键词\n2. 检查公司名称或股票代码是否正确\n3. 确认时间范围是否合适",
    
    "incomplete_data": "找到部分相关信息，但以下数据缺失：\n{missing_items}\n\n已获得信息：\n{available_data}",
    
    "ambiguous_query": "查询需求不够明确。请说明：\n1. 具体的公司名称或股票代码\n2. 需要查询的信息类型\n3. 相关的时间范围",
    
    "multiple_results": "找到多个相关结果。请进一步明确：\n{results_summary}\n\n请指定具体需要哪一项。"
}

# 金融术语词典（中英对照）
FINANCIAL_TERMS = {
    "收购": "Acquisition",
    "合并": "Merger",
    "关连交易": "Connected Transaction", 
    "主要交易": "Major Transaction",
    "每股盈利": "Earnings Per Share (EPS)",
    "市盈率": "Price-to-Earnings Ratio (P/E)",
    "资产净值": "Net Asset Value (NAV)",
    "息税折旧摊销前利润": "EBITDA",
    "尽职调查": "Due Diligence",
    "业绩承诺": "Performance Guarantee",
    "对赌协议": "Valuation Adjustment Mechanism (VAM)",
    "锁定期": "Lock-up Period",
    "公允价值": "Fair Value",
    "商誉": "Goodwill",
    "或有对价": "Contingent Consideration",
    "控股股东": "Controlling Shareholder",
    "独立第三方": "Independent Third Party",
    "备考": "Pro Forma",
    "经审计": "Audited",
    "未经审计": "Unaudited"
}