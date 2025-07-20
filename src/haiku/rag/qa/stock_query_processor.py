import re
from typing import Optional, List, Tuple
from haiku.rag.client import HaikuRAG


class StockQueryProcessor:
    """股票代码查询处理器，使用规则基础方法避免模型幻觉"""
    
    def __init__(self, client: HaikuRAG):
        self.client = client
        
    def is_stock_query(self, question: str) -> bool:
        """判断是否为股票代码查询"""
        # 检测股票代码格式（4-6位数字）
        stock_code_pattern = r'\b\d{4,6}\b'
        
        # 检测查询意图
        query_patterns = [
            r'(\d{4,6}).*是.*公司',
            r'(\d{4,6}).*对应.*公司',
            r'股票代码.*(\d{4,6})',
            r'证券代号.*(\d{4,6})',
            r'(\d{4,6}).*哪.*公司'
        ]
        
        return any(re.search(pattern, question) for pattern in query_patterns)
    
    def extract_stock_code(self, question: str) -> Optional[str]:
        """从问题中提取股票代码"""
        # 寻找4-6位数字
        matches = re.findall(r'\b(\d{4,6})\b', question)
        return matches[0] if matches else None
    
    async def process_stock_query(self, question: str) -> Optional[str]:
        """处理股票代码查询"""
        if not self.is_stock_query(question):
            return None
            
        stock_code = self.extract_stock_code(question)
        if not stock_code:
            return None
            
        # 使用多种查询策略
        search_queries = [
            stock_code,
            f"证券代号 {stock_code}",
            f"股票代码 {stock_code}",
            f"{stock_code} 公司名称"
        ]
        
        best_result = None
        best_score = 0
        
        for query in search_queries:
            results = await self.client.search(query, limit=3)
            for chunk, score in results:
                # 检查是否同时包含股票代码和公司名称
                if stock_code in chunk.content:
                    company_name = self.extract_company_name(chunk.content, stock_code)
                    if company_name and score > best_score:
                        best_result = company_name
                        best_score = score
        
        if best_result:
            return f"根据检索到的文档，股票代码{stock_code}对应的公司是{best_result}。"
        
        return f"根据检索到的文档，我无法找到股票代码{stock_code}对应的公司信息。"
    
    def extract_company_name(self, content: str, stock_code: str) -> Optional[str]:
        """从文档内容中提取公司名称"""
        # 常见的公司名称模式
        patterns = [
            r'公司名稱[:：]\s*([^\s\n]+)',
            r'公司名称[:：]\s*([^\s\n]+)',
            r'(?:致：|向：)\s*([^有限公司]*有限公司)',
            r'([^股份]*股份[^有限公司]*有限公司)',
            r'([^控股]*控股[^有限公司]*有限公司)',
            r'([^投资]*投资[^有限公司]*有限公司)',
            r'([^发展]*发展[^有限公司]*有限公司)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 清理匹配结果
                company_name = match.strip()
                if len(company_name) > 2 and '有限公司' in company_name:
                    return company_name
        
        # 备用：寻找包含特定关键词的公司名称
        lines = content.split('\n')
        for line in lines:
            if stock_code in line and ('公司' in line or '控股' in line):
                # 提取可能的公司名称
                cleaned_line = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', line)
                words = cleaned_line.split()
                for word in words:
                    if len(word) > 4 and ('控股' in word or '有限公司' in word):
                        return word
        
        return None


# 全局实例
_stock_processor = None

def get_stock_processor(client: HaikuRAG) -> StockQueryProcessor:
    """获取股票查询处理器实例"""
    global _stock_processor
    if _stock_processor is None:
        _stock_processor = StockQueryProcessor(client)
    return _stock_processor 