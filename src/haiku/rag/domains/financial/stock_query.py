"""统一的股票代码查询优化器"""

import re
from typing import Optional, List, Tuple
from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import Chunk


class UnifiedStockQueryProcessor:
    """统一的股票代码查询处理器，合并了两个原始模块的最佳功能"""
    
    # 股票代码模式（从两个模块中选择最全面的）
    STOCK_CODE_PATTERNS = [
        r'\b(\d{5})\b',           # 5位数字 (如 00700)
        r'\b(\d{4})\b',            # 4位数字 (如 0700, 1010)
        r'(?:股票代码|股份代號|stock code)[:：\s]*(\d{4,5})',  # 明确的股票代码
        r'(?:代号|代號|code)[:：\s]*(\d{4,5})',               # 简短形式
        r'(?:证券代号|證券代號)[:：\s]*(\d{4,5})',             # 证券代号
    ]
    
    # 查询意图模式
    QUERY_INTENT_PATTERNS = [
        r'(\d{4,6}).*是.*公司',
        r'(\d{4,6}).*对应.*公司',
        r'股票代码.*(\d{4,6})',
        r'证券代号.*(\d{4,6})',
        r'(\d{4,6}).*哪.*公司',
        r'(\d{4,6}).*什么.*公司',
    ]
    
    # 公司名称提取模式
    COMPANY_NAME_PATTERNS = [
        r'公司名[稱称][:：]\s*([^\n]+)',
        r'company name[:：]\s*([^\n]+)',
        r'發行人[:：]\s*([^\n]+)',
        r'issuer[:：]\s*([^\n]+)',
        # 直接在标题中查找
        r'^([^（\(]+(?:有限公司|LIMITED|LTD|COMPANY|CORP))',
        # 额外的模式
        r'(?:致：|向：)\s*([^有限公司]*有限公司)',
        r'([^股份]*股份[^有限公司]*有限公司)',
        r'([^控股]*控股[^有限公司]*有限公司)',
    ]
    
    def __init__(self, client: HaikuRAG):
        self.client = client
    
    def is_stock_query(self, question: str) -> bool:
        """判断是否为股票代码查询"""
        return any(re.search(pattern, question) for pattern in self.QUERY_INTENT_PATTERNS)
    
    def extract_stock_code(self, query: str) -> Optional[str]:
        """从查询中提取股票代码"""
        # 尝试各种模式
        for pattern in self.STOCK_CODE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                code = match.group(1)
                # 补齐到5位（港交所标准）
                if len(code) == 4:
                    code = '0' + code
                return code
        return None
    
    async def search_by_stock_code(
        self, 
        code: str, 
        limit: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """针对股票代码进行精确搜索"""
        # 构建多种查询变体
        search_queries = [
            f'股票代码 {code}',
            f'股份代號 {code}',
            f'证券代号 {code}',
            f'證券代號 {code}',
            f'stock code {code}',
            f'{code}',  # 直接搜索代码
            f"{code} 公司名称",
        ]
        
        all_results = []
        seen_chunks = set()
        
        for query in search_queries:
            results = await self.client.search(query, limit=limit)
            
            for chunk, score in results:
                # 检查内容中是否真的包含股票代码
                if code in chunk.content:
                    # 提高包含精确股票代码的分数
                    boosted_score = score + 0.8
                    
                    # 如果是在股票代码字段附近，进一步提高分数
                    if any(indicator in chunk.content for indicator in [
                        '股票代码', '股份代號', 'stock code', '代码：', '代號：', 
                        '证券代号', '證券代號'
                    ]):
                        boosted_score += 0.5
                    
                    # 避免重复
                    if chunk.id not in seen_chunks:
                        seen_chunks.add(chunk.id)
                        all_results.append((chunk, boosted_score))
        
        # 按分数排序并返回
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:limit]
    
    async def optimize_stock_query(
        self, 
        query: str, 
        limit: int = 5
    ) -> Tuple[List[Tuple[Chunk, float]], Optional[str]]:
        """优化股票查询"""
        # 提取股票代码
        stock_code = self.extract_stock_code(query)
        
        if stock_code:
            # 使用专门的股票代码搜索
            results = await self.search_by_stock_code(stock_code, limit=limit * 2)
            
            # 过滤低相关性结果
            filtered_results = [
                (chunk, score) for chunk, score in results
                if score > 0.3  # 提高相关性阈值
            ]
            
            if filtered_results:
                return filtered_results[:limit], stock_code
            else:
                # 如果没有高相关性结果，返回空
                return [], stock_code
        
        # 如果不是股票代码查询，使用普通搜索
        results = await self.client.search(query, limit=limit)
        return results, None
    
    def extract_company_name(self, content: str, stock_code: str) -> Optional[str]:
        """从文档内容中提取公司名称"""
        # 尝试所有公司名称模式
        for pattern in self.COMPANY_NAME_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                company_name = match.group(1).strip()
                if len(company_name) > 2:
                    return company_name
        
        # 备用：按行搜索包含股票代码和公司关键词的行
        lines = content.split('\n')
        for line in lines:
            if stock_code in line and ('公司' in line or '控股' in line or 'LIMITED' in line):
                # 提取可能的公司名称
                cleaned_line = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', line)
                words = cleaned_line.split()
                for word in words:
                    if len(word) > 4 and any(keyword in word for keyword in ['控股', '有限公司', 'LIMITED', 'LTD']):
                        return word
        
        return None
    
    async def process_stock_query(self, question: str) -> Optional[str]:
        """处理股票代码查询（兼容原接口）"""
        if not self.is_stock_query(question):
            return None
            
        results, stock_code = await self.optimize_stock_query(question)
        
        if not results:
            if stock_code:
                return f"未找到股票代码 {stock_code} 的相关信息。请确认代码是否正确。"
            return None
        
        # 查找包含公司名称的内容
        for chunk, score in results:
            content = chunk.content
            company_name = self.extract_company_name(content, stock_code or "")
            
            if company_name and stock_code and stock_code in content:
                return f"根据文档内容，股票代码 {stock_code} 对应的公司是：{company_name}"
        
        # 如果没有找到公司名称，返回默认信息
        if stock_code:
            return f"根据检索到的文档，找到了股票代码 {stock_code} 的相关信息，但无法明确提取公司名称。"
        
        return None
    
    def format_stock_response(
        self, 
        results: List[Tuple[Chunk, float]], 
        stock_code: Optional[str]
    ) -> Optional[str]:
        """格式化股票查询响应"""
        if not results:
            if stock_code:
                return f"未找到股票代码 {stock_code} 的相关信息。请确认代码是否正确。"
            return None
        
        # 查找包含公司名称的内容
        for chunk, score in results:
            content = chunk.content
            company_name = self.extract_company_name(content, stock_code or "")
            
            if company_name and stock_code and stock_code in content:
                return f"根据文档内容，股票代码 {stock_code} 对应的公司是：{company_name}"
        
        return None


# 全局实例管理
_stock_processor = None

async def get_stock_query_processor(client: HaikuRAG) -> UnifiedStockQueryProcessor:
    """获取统一股票查询处理器实例"""
    global _stock_processor
    if _stock_processor is None:
        _stock_processor = UnifiedStockQueryProcessor(client)
    return _stock_processor