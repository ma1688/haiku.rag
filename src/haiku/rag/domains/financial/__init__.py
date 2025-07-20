"""金融领域专用功能模块

这个包包含所有与金融文档处理相关的功能：
- 金融文档分块器
- 金融领域QA代理
- 股票代码查询处理
- 金融术语和提示词
"""

from haiku.rag.config import Config

# 只有在启用金融功能时才导出相关模块
if Config.USE_FINANCIAL_CHUNKER or Config.USE_FINANCIAL_QA:
    from .chunker import FinancialChunker
    from .qa import FinancialQuestionAnswerAgent
    from .stock_query import UnifiedStockQueryProcessor
    
    __all__ = [
        'FinancialChunker',
        'FinancialQuestionAnswerAgent', 
        'UnifiedStockQueryProcessor'
    ]
else:
    __all__ = []