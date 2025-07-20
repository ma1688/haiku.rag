"""金融领域专用问答代理"""

import json
from typing import Dict, List, Tuple, Optional
from collections.abc import Sequence

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.base import QuestionAnswerAgentBase
from .prompts import (
    FINANCIAL_SYSTEM_PROMPT,
    get_intent_prompt,
    ANSWER_FORMATS,
    ERROR_MESSAGES,
    FINANCIAL_TERMS
)
from haiku.rag.store.models.chunk import Chunk


class FinancialQuestionAnswerAgent(QuestionAnswerAgentBase):
    """金融领域专用的问答代理，优化港交所公告查询"""
    
    def __init__(self, client: HaikuRAG):
        """初始化金融问答代理"""
        super().__init__(client)
        self._system_prompt = FINANCIAL_SYSTEM_PROMPT
        self._enhanced_tools = self._create_enhanced_tools()
    
    def _create_enhanced_tools(self) -> list:
        """创建增强的金融查询工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_financial_documents",
                    "description": "搜索金融公告文档，支持多种过滤条件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询词（支持公司名、股票代码、交易类型等）",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "返回结果数量限制",
                                "default": 5,
                            },
                            "announcement_type": {
                                "type": "string",
                                "description": "公告类型过滤",
                                "enum": ["earnings", "acquisition", "connected_transaction", "all"],
                                "default": "all"
                            },
                            "date_range": {
                                "type": "object",
                                "description": "日期范围过滤",
                                "properties": {
                                    "start_date": {"type": "string", "format": "date"},
                                    "end_date": {"type": "string", "format": "date"}
                                }
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_financial_data",
                    "description": "从文档中提取结构化的财务数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_content": {
                                "type": "string",
                                "description": "要分析的文档内容",
                            },
                            "data_types": {
                                "type": "array",
                                "description": "需要提取的数据类型",
                                "items": {
                                    "type": "string",
                                    "enum": ["amounts", "ratios", "dates", "parties", "terms"]
                                }
                            }
                        },
                        "required": ["document_content"],
                    },
                },
            }
        ]
    
    async def _detect_query_intent(self, question: str) -> str:
        """检测查询意图"""
        # 使用 financial_prompts 中的意图识别
        intent_prompt = get_intent_prompt(question)
        return intent_prompt
    
    async def _format_financial_data(self, data: Dict) -> str:
        """格式化财务数据输出"""
        # 根据数据类型选择合适的格式模板
        if "transaction_elements" in data:
            template = ANSWER_FORMATS["transaction_summary"]
        elif "financial_data" in data:
            template = ANSWER_FORMATS["financial_data"]
        elif "compliant_items" in data:
            template = ANSWER_FORMATS["compliance_report"]
        elif "comparison_table" in data:
            template = ANSWER_FORMATS["comparison_table"]
        else:
            # 默认格式
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        # 填充模板
        try:
            return template.format(**data)
        except KeyError:
            # 如果某些字段缺失，返回原始数据
            return json.dumps(data, ensure_ascii=False, indent=2)
    
    async def _enhance_search_results(
        self, 
        results: List[Tuple[Chunk, float]], 
        query: str
    ) -> List[Dict]:
        """增强搜索结果，添加结构化信息"""
        enhanced_results = []
        
        for chunk, score in results:
            # 提取元数据
            metadata = chunk.metadata or {}
            
            # 识别公告类型
            content_lower = chunk.content.lower()
            announcement_type = "general"
            if "盈利" in content_lower or "earnings" in content_lower:
                announcement_type = "earnings"
            elif "收购" in content_lower or "acquisition" in content_lower:
                announcement_type = "acquisition"
            elif "关连交易" in content_lower or "connected transaction" in content_lower:
                announcement_type = "connected_transaction"
            
            enhanced_results.append({
                "content": chunk.content,
                "score": score,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "announcement_type": announcement_type,
                "metadata": metadata
            })
        
        return enhanced_results
    
    async def _extract_key_information(self, content: str, query_type: str) -> Dict:
        """从内容中提取关键信息"""
        extracted_info = {}
        
        # 根据查询类型提取不同信息
        if "data" in query_type or "financial" in query_type:
            # 提取金额
            import re
            amounts = re.findall(
                r'(?:港币|HK\$|人民币|RMB|美元|USD)\s*([\d,]+(?:\.\d+)?)\s*(?:百万|千万|亿|元|million|billion)?',
                content
            )
            if amounts:
                extracted_info["amounts"] = amounts
            
            # 提取百分比
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
            if percentages:
                extracted_info["percentages"] = percentages
            
            # 提取日期
            dates = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2})', content)
            if dates:
                extracted_info["dates"] = dates
        
        elif "transaction" in query_type:
            # 提取交易方
            parties_pattern = r'(?:买方|卖方|收购方|出售方|Buyer|Seller)[:：]\s*([^\n]+)'
            parties = re.findall(parties_pattern, content, re.IGNORECASE)
            if parties:
                extracted_info["parties"] = parties
            
            # 提取交易条件
            conditions_keywords = ["先决条件", "条件", "Conditions", "Prerequisites"]
            for keyword in conditions_keywords:
                if keyword in content:
                    # 简单提取关键词后的内容
                    idx = content.find(keyword)
                    if idx != -1:
                        extracted_info["conditions_context"] = content[idx:idx+500]
                        break
        
        return extracted_info
    
    async def answer(self, question: str) -> str:
        """回答金融相关问题"""
        try:
            # 1. 检测查询意图
            intent_prompt = await self._detect_query_intent(question)
            
            # 2. 构建增强的系统提示词
            enhanced_prompt = self._system_prompt
            if intent_prompt:
                enhanced_prompt += "\n\n" + intent_prompt
            
            # 3. 执行搜索
            search_results = await self._client.search(question, limit=5)
            
            if not search_results:
                return ERROR_MESSAGES["no_data_found"].format(query_topic=question)
            
            # 4. 增强搜索结果
            enhanced_results = await self._enhance_search_results(search_results, question)
            
            # 5. 提取关键信息
            all_extracted_info = {}
            for result in enhanced_results:
                extracted = await self._extract_key_information(
                    result["content"], 
                    question.lower()
                )
                if extracted:
                    all_extracted_info.update(extracted)
            
            # 6. 构建上下文
            context_parts = []
            for i, result in enumerate(enhanced_results, 1):
                context_parts.append(
                    f"【文档 {i}】\n"
                    f"类型：{result['announcement_type']}\n"
                    f"相关度：{result['score']:.4f}\n"
                    f"内容：{result['content']}\n"
                )
            
            context = "\n".join(context_parts)
            
            # 7. 生成结构化回答
            response_data = {
                "query": question,
                "intent": intent_prompt[:50] if intent_prompt else "general",
                "results_count": len(enhanced_results),
                "extracted_info": all_extracted_info,
                "context": context,
                "source_reference": f"基于 {len(enhanced_results)} 份相关文档"
            }
            
            # 8. 格式化输出
            if all_extracted_info:
                # 如果成功提取了结构化信息
                formatted_response = await self._format_financial_data({
                    "financial_data": json.dumps(all_extracted_info, ensure_ascii=False, indent=2),
                    "source_reference": response_data["source_reference"],
                    "company_name": "相关公司",  # 可以从context中提取
                    "announcement_type": enhanced_results[0]["announcement_type"] if enhanced_results else "general"
                })
                return formatted_response
            else:
                # 返回原始上下文供进一步分析
                return f"""
根据检索到的 {len(enhanced_results)} 份文档，以下是相关信息：

{context}

请注意：以上信息来自文档原文，如需更详细的分析，请提出具体问题。
"""
            
        except Exception as e:
            return f"处理查询时出现错误：{str(e)}\n\n{ERROR_MESSAGES['ambiguous_query']}"


class FinancialQuestionAnswerOpenAIAgent(FinancialQuestionAnswerAgent):
    """集成 OpenAI API 的金融问答代理"""
    
    def __init__(self, client: HaikuRAG, model: str = "gpt-4o"):
        super().__init__(client)
        self._model = model
        self._init_openai_client()
    
    def _init_openai_client(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import AsyncOpenAI
            from haiku.rag.config import Config
            
            self.openai_client = AsyncOpenAI(
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_BASE_URL if Config.OPENAI_BASE_URL else None
            )
            
            # 转换工具格式
            from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
            self.tools: Sequence[ChatCompletionToolParam] = [
                ChatCompletionToolParam(tool) for tool in self._enhanced_tools
            ]
        except ImportError:
            raise ImportError("请安装 openai 包：pip install openai")
    
    async def answer(self, question: str) -> str:
        """使用 OpenAI 增强的金融问答"""
        from openai.types.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionToolMessageParam,
            ChatCompletionUserMessageParam,
        )
        
        # 检测意图并构建提示词
        intent_prompt = await self._detect_query_intent(question)
        system_prompt = self._system_prompt
        if intent_prompt:
            system_prompt += "\n\n" + intent_prompt
        
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=question),
        ]
        
        max_rounds = 3
        
        for _ in range(max_rounds):
            response = await self.openai_client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self.tools,
                temperature=0.1,  # 低温度以获得更一致的金融数据
            )
            
            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=response_message.content,
                        tool_calls=[
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ],
                    )
                )
                
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "search_financial_documents":
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query", question)
                        limit = int(args.get("limit", 5))
                        
                        # 执行搜索
                        search_results = await self._client.search(query, limit=limit)
                        enhanced_results = await self._enhance_search_results(
                            search_results, query
                        )
                        
                        # 构建工具响应
                        tool_response = {
                            "results": enhanced_results,
                            "total": len(enhanced_results)
                        }
                        
                        messages.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                content=json.dumps(tool_response, ensure_ascii=False),
                                tool_call_id=tool_call.id,
                            )
                        )
                    
                    elif tool_call.function.name == "extract_financial_data":
                        args = json.loads(tool_call.function.arguments)
                        content = args.get("document_content", "")
                        
                        # 提取财务数据
                        extracted_data = await self._extract_key_information(
                            content, question.lower()
                        )
                        
                        messages.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                content=json.dumps(extracted_data, ensure_ascii=False),
                                tool_call_id=tool_call.id,
                            )
                        )
            else:
                # 没有工具调用，返回最终答案
                return response_message.content or "无法生成回答"
        
        return "查询处理超时，请尝试简化问题或分步查询。"


class FinancialQuestionAnswerOllamaAgent(FinancialQuestionAnswerAgent):
    """集成 Ollama 的金融问答代理"""
    
    def __init__(self, client: HaikuRAG, model: str = "qwen2.5:14b"):
        super().__init__(client)
        self._model = model
    
    async def answer(self, question: str) -> str:
        """使用 Ollama 的金融问答（简化版）"""
        # 执行基础搜索
        search_results = await self._client.search(question, limit=5)
        
        if not search_results:
            return ERROR_MESSAGES["no_data_found"].format(query_topic=question)
        
        # 增强搜索结果
        enhanced_results = await self._enhance_search_results(search_results, question)
        
        # 构建上下文
        context_parts = []
        for i, result in enumerate(enhanced_results, 1):
            context_parts.append(f"文档{i}：{result['content'][:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # 使用 Ollama 生成回答
        try:
            from haiku.rag.qa.ollama import QuestionAnswerOllamaAgent
            
            # 创建临时的 Ollama 代理并设置金融提示词
            temp_agent = QuestionAnswerOllamaAgent(self._client, self._model)
            temp_agent._system_prompt = self._system_prompt + "\n\n" + await self._detect_query_intent(question)
            
            # 构建带上下文的问题
            contextualized_question = f"""
基于以下文档内容回答问题：

{context}

问题：{question}
"""
            
            return await temp_agent.answer(contextualized_question)
            
        except Exception as e:
            # 如果 Ollama 不可用，返回基础结果
            return await super().answer(question)