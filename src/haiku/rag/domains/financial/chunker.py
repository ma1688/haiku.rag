import re
from typing import Optional, List, Dict, Tuple

from haiku.rag.chunker import Chunker


class FinancialChunker(Chunker):
    """增强的金融文档分块工具，特别适用于香港交易所公告。
    
    此分块工具扩展了基础 Chunker，特别处理以下内容：
    - 部分标题和文档结构
    - 表格和财务数据
    - 法律条款和编号列表
    - 香港文档中常见的中英文混合内容
    """

    # 香港交易所公告的部分标题模式
    SECTION_PATTERNS = [r'^[\s]*(\d+\.[\d\.]*)\s+(.+)$',  # 编号部分（1. 2. 1.1等）
        r'^[\s]*([一二三四五六七八九十]+[、.])\s*(.+)$', r'^[\s]*[(（][一二三四五六七八九十]+[)）]\s*(.+)$',
        # 字母部分（A. B. (a)等）
        r'^[\s]*([A-Z]\.)\s+(.+)$', r'^[\s]*[(（][a-zA-Z][)）]\s*(.+)$', # 特殊关键字
        r'^[\s]*(背景|BACKGROUND|交易|TRANSACTION|財務影響|FINANCIAL IMPACT|風險|RISK)[:：]?\s*$',
        r'^[\s]*(董事會|BOARD|建議|RECOMMENDATION|條款|TERMS|先決條件|CONDITIONS PRECEDENT)[:：]?\s*$', ]

    # 表格指示器
    TABLE_INDICATORS = [r'[-─━]+\s*[-─━]+',  # 表格边框
        r'\|.*\|.*\|',  # 管道分隔的表格
        r'^\s*[^\s]+\s+\d+[,，]\d+',  # 财务数字对齐
        r'人民幣|RMB|HK\$|港元|USD|美元',  # 货币指示器
    ]

    # 重要的财务术语，不应被分割
    FINANCIAL_TERMS = ['每股', '市盈率', '資產淨值', '股息', '派息', '配股', '供股', 'earnings per share', 'P/E ratio',
        'NAV', 'dividend', 'rights issue', '收購', '出售', '交易對價', '代價', 'acquisition', 'disposal', '關連交易',
        'connected transaction', '主要交易', 'major transaction']

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100,
                 min_chunk_size: int = 300, max_chunk_size: int = 500,
                 chunk_size_variance: int = 100, preserve_tables: bool = True,
                 extract_metadata: bool = True):
        """初始化金融分块器，带有动态切块大小参数。

        参数:
            chunk_size: 目标块大小（以标记为单位，默认500）
            chunk_overlap: 块之间的重叠大小（默认100）
            min_chunk_size: 最小块大小（默认300）
            max_chunk_size: 最大块大小（默认500）
            chunk_size_variance: 块大小浮动范围（默认100）
            preserve_tables: 是否避免分割表格
            extract_metadata: 是否提取文档元数据
        """
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_size_variance = chunk_size_variance
        self.preserve_tables = preserve_tables
        self.extract_metadata = extract_metadata
        self._section_headers: List[Tuple[int, str]] = []

    def _detect_section_header(self, line: str) -> Optional[str]:
        """检测一行是否为部分标题。"""
        line = line.strip()
        if not line:
            return None

        for pattern in self.SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return line
        return None

    def _is_in_table(self, text: str, position: int, window: int = 200) -> bool:
        """检查指定位置是否在表格内。"""
        if not self.preserve_tables:
            return False

        # 在位置周围查看表格指示器
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]

        for indicator in self.TABLE_INDICATORS:
            if re.search(indicator, context):
                return True
        return False

    def _contains_financial_term(self, text: str, position: int, window: int = 50) -> bool:
        """检查指定位置是否靠近重要的财务术语。"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()

        for term in self.FINANCIAL_TERMS:
            if term.lower() in context:
                return True
        return False

    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """从文档头部提取元数据。"""
        metadata = {}
        lines = text.split('\n')[:20]  # 检查前20行

        for line in lines:
            # 股票代码
            stock_code_match = re.search(r'股份代號[：:]\s*(\d+)', line, re.IGNORECASE)
            if not stock_code_match:
                stock_code_match = re.search(r'Stock Code[：:]\s*(\d+)', line, re.IGNORECASE)
            if not stock_code_match:
                stock_code_match = re.search(r'股票代碼[：:]\s*(\d+)', line, re.IGNORECASE)

            if stock_code_match:
                metadata['stock_code'] = stock_code_match.group(1).strip()

            # 公司名称
            company_match = re.search(r'公司名稱[：:]\s*(.+)', line, re.IGNORECASE)
            if not company_match:
                company_match = re.search(r'Company Name[：:]\s*(.+)', line, re.IGNORECASE)

            if company_match:
                metadata['company_name'] = company_match.group(1).strip()

            # 公告类型
            if '盈利' in line or 'earnings' in line.lower():
                metadata['type'] = 'earnings'
            elif '收購' in line or 'acquisition' in line.lower():
                metadata['type'] = 'acquisition'
            elif '關連交易' in line or 'connected transaction' in line.lower():
                metadata['type'] = 'connected_transaction'

        return metadata

    def _preprocess_text(self, text: str) -> str:
        """金融文档的增强预处理。"""
        # 首先应用基础预处理
        text = super()._preprocess_text(text)

        # 对金融文档进行额外预处理
        # 通过不规范化表格空格来保留表格结构
        lines = text.split('\n')
        processed_lines = []
        in_table = False

        for line in lines:
            # 侦测表格的开始/结束
            if any(re.search(pattern, line) for pattern in self.TABLE_INDICATORS):
                in_table = True
            elif in_table and line.strip() == '':
                in_table = False

            if in_table:
                # 在表格中保留原始间距
                processed_lines.append(line)
            else:
                # 非表格文本的常规处理
                processed_lines.append(line.strip())

        # 提取部分标题以保留结构
        self._section_headers = []
        for i, line in enumerate(processed_lines):
            header = self._detect_section_header(line)
            if header:
                self._section_headers.append((i, header))

        return '\n'.join(processed_lines)

    def _find_best_split_point(self, text: str, target_pos: int) -> int:
        """金融文档的增强分割点检测。"""
        # 扩大金融文档的搜索窗口
        search_window = min(200, len(text) // 3)

        start_search = max(0, target_pos - search_window)
        end_search = min(len(text), target_pos + search_window)

        best_pos = target_pos
        best_score = 0

        # 部分标题具有最高优先级
        for i in range(start_search, end_search):
            score = 0

            # 检查是否在部分标题之前
            remaining_text = text[i:]
            first_line = remaining_text.split('\n')[0] if '\n' in remaining_text else remaining_text
            if self._detect_section_header(first_line):
                score = 15  # 最高优先级

            # 避免在表格中分割
            elif self._is_in_table(text, i):
                score = -10  # 避免负分

            # 避免在财务术语附近分割
            elif self._contains_financial_term(text, i):
                score = -5

            # 标准句子结尾（来自父类）
            elif i < len(text) and text[i] in ['。', '！', '？', '；']:
                score = 10
            elif i < len(text) - 1 and text[i:i + 2] in ['. ', '! ', '? ', '; ']:
                score = 10

            # 段落换行
            elif i < len(text) - 1 and text[i:i + 2] == '\n\n':
                score = 12  # 财务文档优先级高于句子

            # 单行换行（通常用于列表）
            elif i < len(text) and text[i] == '\n':
                score = 8

            # 编号列表项
            elif i < len(text) - 2 and re.match(r'\n\d+[\.)]\s', text[i:i + 4]):
                score = 9

            # 对逗号的优先级较低
            elif i < len(text) and text[i] in ['，', ',']:
                score = 2

            # 优先考虑距离目标更近的位置
            distance_penalty = abs(i - target_pos) / search_window
            final_score = score * (1 - distance_penalty * 0.5)  # 财务文档的惩罚较小

            if final_score > best_score:
                best_score = final_score
                best_pos = i + 1 if score > 0 else i

        return best_pos

    def _get_dynamic_chunk_size(self, text_position: int, total_length: int) -> int:
        """根据文本位置和内容动态计算切块大小。

        参数:
            text_position: 当前文本位置
            total_length: 文本总长度

        返回:
            动态计算的切块大小
        """
        # 基础大小在min和max之间
        base_size = (self.min_chunk_size + self.max_chunk_size) // 2

        # 根据文档位置调整（开头和结尾使用较小的块）
        position_ratio = text_position / total_length if total_length > 0 else 0
        if position_ratio < 0.1 or position_ratio > 0.9:
            # 文档开头和结尾使用较小的块
            size_adjustment = -self.chunk_size_variance // 2
        else:
            # 文档中间使用较大的块
            size_adjustment = self.chunk_size_variance // 2

        dynamic_size = base_size + size_adjustment

        # 确保在范围内
        return max(self.min_chunk_size, min(self.max_chunk_size, dynamic_size))

    async def _dynamic_chunk_text(self, text: str) -> List[str]:
        """使用动态切块大小进行文本分块。"""
        if not text:
            return []

        # 编码文本为 tokens 列表
        encoded_tokens = self.encoder.encode(text, disallowed_special=())

        if len(encoded_tokens) <= self.max_chunk_size:
            return [text]  # 如果文本很短，直接返回

        chunks = []
        i = 0

        while i < len(encoded_tokens):
            # 动态计算当前块的大小
            current_chunk_size = self._get_dynamic_chunk_size(i, len(encoded_tokens))

            start_i = i
            end_i = min(i + current_chunk_size, len(encoded_tokens))

            # 尝试找到更好的分割点
            if end_i < len(encoded_tokens):
                chunk_tokens = encoded_tokens[start_i:end_i]
                temp_text = self.encoder.decode(chunk_tokens)

                # 找到更好的分割点
                better_end = self._find_best_split_point(temp_text, len(temp_text))

                if better_end < len(temp_text):
                    better_text = temp_text[:better_end]
                    better_tokens = self.encoder.encode(better_text, disallowed_special=())
                    end_i = start_i + len(better_tokens)

            # 解码当前块的 tokens
            chunk_tokens = encoded_tokens[start_i:end_i]
            chunk_text = self.encoder.decode(chunk_tokens).strip()

            if chunk_text:  # 只添加非空的块
                chunks.append(chunk_text)

            # 如果到达最后的可能块，则退出循环
            if end_i >= len(encoded_tokens):
                break

            # 处理重叠部分
            i += max(1, end_i - start_i - self.chunk_overlap)

        return chunks

    async def chunk(self, text: str, return_metadata: bool = False) -> List[str] | Tuple[List[str], Dict]:
        """使用动态切块大小和增强结构保留分块金融文档。

        参数:
            text: 要分块的文档文本
            return_metadata: 是否返回提取的元数据

        返回:
            分块列表，可能带有元数据字典
        """
        if not text:
            return ([], {}) if return_metadata else []

        # 如果请求，提取元数据
        metadata = self._extract_document_metadata(text) if self.extract_metadata else {}

        # 使用结构检测进行预处理
        text = self._preprocess_text(text)

        # 使用动态切块
        chunks = await self._dynamic_chunk_text(text)

        # 后处理块以添加部分上下文
        enhanced_chunks = []
        for chunk in chunks:
            # 找到此块属于哪一部分
            chunk_start = text.find(chunk)
            current_section = None

            for pos, header in reversed(self._section_headers):
                if pos <= chunk_start:
                    current_section = header
                    break

            # 如果上下文尚未存在，则添加部分上下文
            if current_section and not chunk.startswith(current_section):
                # 仅在不使块过大时添加
                with_context = f"{current_section}\n\n{chunk}"
                encoded = self.encoder.encode(with_context)
                if len(encoded) <= self.max_chunk_size * 1.1:  # 允许10%的溢出
                    chunk = with_context

            enhanced_chunks.append(chunk)

        if return_metadata:
            return enhanced_chunks, metadata
        return enhanced_chunks


# 便利的实例
financial_chunker = FinancialChunker()
