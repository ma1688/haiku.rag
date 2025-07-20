"""测试金融领域专用问答系统"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from haiku.rag.qa.financial_prompts import (
    get_intent_prompt,
    FINANCIAL_TERMS,
    ANSWER_FORMATS,
    ERROR_MESSAGES
)
from haiku.rag.qa.financial_qa import (
    FinancialQuestionAnswerAgent,
    FinancialQuestionAnswerOpenAIAgent
)
from haiku.rag.store.models.chunk import Chunk


class TestFinancialPrompts:
    """测试金融提示词系统"""
    
    def test_intent_detection(self):
        """测试查询意图识别"""
        # 数据提取类
        assert "FINANCIAL_DATA_PROMPT" in get_intent_prompt("腾讯的营收是多少？")
        assert "FINANCIAL_DATA_PROMPT" in get_intent_prompt("What is the profit margin?")
        
        # 交易分析类
        assert "TRANSACTION_ANALYSIS_PROMPT" in get_intent_prompt("分析这次收购的条款")
        assert "TRANSACTION_ANALYSIS_PROMPT" in get_intent_prompt("Tell me about the acquisition terms")
        
        # 合规检查类
        assert "COMPLIANCE_CHECK_PROMPT" in get_intent_prompt("这是否构成关连交易？")
        assert "COMPLIANCE_CHECK_PROMPT" in get_intent_prompt("Check compliance requirements")
        
        # 比较分析类
        assert "COMPARATIVE_ANALYSIS_PROMPT" in get_intent_prompt("比较两家公司的估值")
        assert "COMPARATIVE_ANALYSIS_PROMPT" in get_intent_prompt("Compare the valuation")
        
        # 默认情况
        assert get_intent_prompt("随便问点什么") == ""
    
    def test_financial_terms_coverage(self):
        """测试金融术语词典覆盖度"""
        essential_terms = [
            "收购", "关连交易", "每股盈利", "市盈率",
            "尽职调查", "业绩承诺", "公允价值"
        ]
        
        for term in essential_terms:
            assert term in FINANCIAL_TERMS
            assert FINANCIAL_TERMS[term]  # 确保有英文对照
    
    def test_answer_formats(self):
        """测试回答格式模板"""
        # 确保所有格式都包含必要占位符
        for format_name, template in ANSWER_FORMATS.items():
            assert "{source_reference}" in template
            
        # 测试格式化
        test_data = {
            "company_name": "测试公司",
            "announcement_type": "收购公告",
            "financial_data": "营收：100亿",
            "key_dates": "2024-01-15",
            "data_notes": "经审计",
            "source_reference": "文档1"
        }
        
        formatted = ANSWER_FORMATS["financial_data"].format(**test_data)
        assert "测试公司" in formatted
        assert "100亿" in formatted


class TestFinancialQuestionAnswerAgent:
    """测试金融问答代理"""
    
    @pytest.fixture
    def mock_client(self):
        """创建模拟客户端"""
        client = MagicMock()
        client.search = AsyncMock()
        return client
    
    @pytest.fixture
    def agent(self, mock_client):
        """创建金融问答代理"""
        return FinancialQuestionAnswerAgent(mock_client)
    
    @pytest.mark.asyncio
    async def test_intent_detection_in_agent(self, agent):
        """测试代理中的意图检测"""
        intent = await agent._detect_query_intent("腾讯的营收是多少？")
        assert "财务信息" in intent or "FINANCIAL_DATA" in intent
    
    @pytest.mark.asyncio
    async def test_enhance_search_results(self, agent):
        """测试搜索结果增强"""
        # 创建模拟搜索结果
        mock_chunks = [
            (Chunk(id=1, document_id=1, content="盈利公告：营收100亿", metadata={}), 0.9),
            (Chunk(id=2, document_id=1, content="收购XYZ公司，代价50亿", metadata={}), 0.8),
        ]
        
        enhanced = await agent._enhance_search_results(mock_chunks, "营收")
        
        assert len(enhanced) == 2
        assert enhanced[0]["announcement_type"] == "earnings"
        assert enhanced[1]["announcement_type"] == "acquisition"
        assert enhanced[0]["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_extract_key_information(self, agent):
        """测试关键信息提取"""
        content = """
        本公司宣布以港币50亿元收购ABC公司100%股权。
        交易将带来15%的投资回报率。
        交易预计于2024年3月15日完成。
        """
        
        extracted = await agent._extract_key_information(content, "financial")
        
        assert "amounts" in extracted
        assert any("50" in amount for amount in extracted["amounts"])
        assert "percentages" in extracted
        assert "15" in extracted["percentages"]
        assert "dates" in extracted
        assert any("2024" in date for date in extracted["dates"])
    
    @pytest.mark.asyncio
    async def test_format_financial_data(self, agent):
        """测试财务数据格式化"""
        data = {
            "financial_data": '{"amounts": ["50亿"], "percentages": ["15"]}',
            "source_reference": "基于2份文档",
            "company_name": "ABC公司",
            "announcement_type": "acquisition"
        }
        
        formatted = await agent._format_financial_data(data)
        
        assert "ABC公司" in formatted
        assert "50亿" in formatted
        assert "基于2份文档" in formatted
    
    @pytest.mark.asyncio
    async def test_answer_with_no_results(self, agent, mock_client):
        """测试无结果时的回答"""
        mock_client.search.return_value = []
        
        answer = await agent.answer("不存在的公司信息")
        
        assert "未找到" in answer or "no_data_found" in answer
    
    @pytest.mark.asyncio
    async def test_answer_with_results(self, agent, mock_client):
        """测试有结果时的回答"""
        # 模拟搜索结果
        mock_client.search.return_value = [
            (Chunk(
                id=1, 
                document_id=1, 
                content="腾讯控股有限公司（股票代码：0700）宣布2023年营收为6000亿元", 
                metadata={}
            ), 0.95)
        ]
        
        answer = await agent.answer("腾讯的营收是多少？")
        
        assert "6000亿" in answer or "0700" in answer
        assert "基于" in answer  # 应该包含来源信息
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_client):
        """测试错误处理"""
        mock_client.search.side_effect = Exception("搜索服务不可用")
        
        answer = await agent.answer("查询测试")
        
        assert "错误" in answer or "Error" in answer


class TestFinancialQuestionAnswerOpenAIAgent:
    """测试 OpenAI 集成的金融问答代理"""
    
    @pytest.fixture
    def mock_client(self):
        """创建模拟客户端"""
        client = MagicMock()
        client.search = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_openai(self, monkeypatch):
        """模拟 OpenAI 客户端"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="基于文档分析，营收为100亿元",
                    tool_calls=None
                )
            )
        ]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # 模拟 AsyncOpenAI 类
        mock_openai_class = MagicMock(return_value=mock_client)
        
        # 创建模拟模块
        import sys
        from types import ModuleType
        
        mock_openai_module = ModuleType('openai')
        mock_openai_module.AsyncOpenAI = mock_openai_class
        
        # 注入到 sys.modules
        sys.modules['openai'] = mock_openai_module
        
        # 同时模拟类型导入
        mock_types = ModuleType('openai.types.chat')
        mock_types.ChatCompletionAssistantMessageParam = dict
        mock_types.ChatCompletionMessageParam = dict
        mock_types.ChatCompletionSystemMessageParam = dict
        mock_types.ChatCompletionToolMessageParam = dict
        mock_types.ChatCompletionUserMessageParam = dict
        
        sys.modules['openai.types'] = ModuleType('openai.types')
        sys.modules['openai.types.chat'] = mock_types
        
        mock_tool_param = ModuleType('openai.types.chat.chat_completion_tool_param')
        mock_tool_param.ChatCompletionToolParam = dict
        sys.modules['openai.types.chat.chat_completion_tool_param'] = mock_tool_param
        
        return mock_client
    
    @pytest.mark.asyncio
    async def test_openai_agent_initialization(self, mock_client, mock_openai, monkeypatch):
        """测试 OpenAI 代理初始化"""
        # 模拟配置
        monkeypatch.setattr("haiku.rag.config.Config.OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("haiku.rag.config.Config.OPENAI_BASE_URL", "")
        
        agent = FinancialQuestionAnswerOpenAIAgent(mock_client, "gpt-4")
        
        assert agent._model == "gpt-4"
        assert hasattr(agent, "openai_client")
        assert hasattr(agent, "tools")
    
    @pytest.mark.asyncio
    async def test_openai_agent_answer(self, mock_client, mock_openai, monkeypatch):
        """测试 OpenAI 代理回答"""
        # 模拟配置
        monkeypatch.setattr("haiku.rag.config.Config.OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("haiku.rag.config.Config.OPENAI_BASE_URL", "")
        
        agent = FinancialQuestionAnswerOpenAIAgent(mock_client, "gpt-4")
        
        answer = await agent.answer("腾讯的营收是多少？")
        
        assert "100亿" in answer
        assert mock_openai.chat.completions.create.called


def test_financial_prompts_completeness():
    """测试金融提示词的完整性"""
    from haiku.rag.qa import financial_prompts
    
    # 确保所有必要的导出都存在
    assert hasattr(financial_prompts, "FINANCIAL_SYSTEM_PROMPT")
    assert hasattr(financial_prompts, "FINANCIAL_DATA_PROMPT")
    assert hasattr(financial_prompts, "TRANSACTION_ANALYSIS_PROMPT")
    assert hasattr(financial_prompts, "COMPLIANCE_CHECK_PROMPT")
    assert hasattr(financial_prompts, "COMPARATIVE_ANALYSIS_PROMPT")
    assert hasattr(financial_prompts, "get_intent_prompt")
    assert hasattr(financial_prompts, "ANSWER_FORMATS")
    assert hasattr(financial_prompts, "ERROR_MESSAGES")
    assert hasattr(financial_prompts, "FINANCIAL_TERMS")