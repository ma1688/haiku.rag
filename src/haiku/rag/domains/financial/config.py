"""金融领域专用配置"""

from haiku.rag.config import Config


class FinancialConfig:
    """金融领域配置管理"""
    
    # 从主配置继承的金融设置
    USE_FINANCIAL_CHUNKER = Config.USE_FINANCIAL_CHUNKER
    USE_FINANCIAL_QA = Config.USE_FINANCIAL_QA
    FINANCIAL_CHUNK_SIZE = Config.FINANCIAL_CHUNK_SIZE
    FINANCIAL_CHUNK_OVERLAP = Config.FINANCIAL_CHUNK_OVERLAP
    PRESERVE_TABLES = Config.PRESERVE_TABLES
    EXTRACT_METADATA = Config.EXTRACT_METADATA
    
    # 金融领域特定设置
    FINANCIAL_TERMS_DICT = {
        "收购": "Acquisition",
        "合并": "Merger", 
        "关连交易": "Connected Transaction",
        "主要交易": "Major Transaction",
        "股东大会": "Annual General Meeting",
        "年度报告": "Annual Report",
        "财务报表": "Financial Statement",
        "董事会": "Board of Directors",
        "监事会": "Supervisory Board",
        "审计": "Audit",
        "投资": "Investment",
        "利润": "Profit",
        "营收": "Revenue",
        "资产": "Assets",
        "负债": "Liabilities",
        "现金流": "Cash Flow",
        "股价": "Stock Price",
        "分红": "Dividend",
        "配股": "Rights Issue"
    }
    
    # 股票代码配置
    STOCK_CODE_MIN_LENGTH = 4
    STOCK_CODE_MAX_LENGTH = 6
    STOCK_CODE_NORMALIZE = True  # 是否将4位代码补齐为5位
    
    # 搜索相关性阈值
    STOCK_SEARCH_SCORE_THRESHOLD = 0.3
    STOCK_SEARCH_BOOST_EXACT_MATCH = 0.8
    STOCK_SEARCH_BOOST_CONTEXT_MATCH = 0.5
    
    @classmethod
    def is_enabled(cls) -> bool:
        """检查是否启用了任何金融功能"""
        return cls.USE_FINANCIAL_CHUNKER or cls.USE_FINANCIAL_QA
    
    @classmethod
    def get_search_config(cls) -> dict:
        """获取搜索相关配置"""
        return {
            'score_threshold': cls.STOCK_SEARCH_SCORE_THRESHOLD,
            'boost_exact_match': cls.STOCK_SEARCH_BOOST_EXACT_MATCH,
            'boost_context_match': cls.STOCK_SEARCH_BOOST_CONTEXT_MATCH,
            'normalize_codes': cls.STOCK_CODE_NORMALIZE
        }
    
    @classmethod
    def get_chunker_config(cls) -> dict:
        """获取分块器配置"""
        return {
            'chunk_size': cls.FINANCIAL_CHUNK_SIZE,
            'chunk_overlap': cls.FINANCIAL_CHUNK_OVERLAP,
            'preserve_tables': cls.PRESERVE_TABLES,
            'extract_metadata': cls.EXTRACT_METADATA
        }