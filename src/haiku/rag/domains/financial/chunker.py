"""Financial document chunker optimized for Hong Kong Exchange announcements."""

import re
from typing import ClassVar, Optional, List, Dict, Tuple
import tiktoken
from haiku.rag.config import Config
from haiku.rag.chunker import Chunker


class FinancialChunker(Chunker):
    """Enhanced chunker for financial documents, especially HKEx announcements.
    
    This chunker extends the base Chunker with special handling for:
    - Section headers and document structure
    - Tables and financial data
    - Legal clauses and numbered lists
    - Chinese/English mixed content common in HK documents
    """
    
    # Section header patterns for HKEx announcements
    SECTION_PATTERNS = [
        # Numbered sections (1. 2. 1.1 etc)
        r'^[\s]*(\d+\.[\d\.]*)\s+(.+)$',
        # Chinese numbered sections (一、二、(一) etc)
        r'^[\s]*([一二三四五六七八九十]+[、.])\s*(.+)$',
        r'^[\s]*[(（][一二三四五六七八九十]+[)）]\s*(.+)$',
        # Letter sections (A. B. (a) etc)
        r'^[\s]*([A-Z]\.)\s+(.+)$',
        r'^[\s]*[(（][a-zA-Z][)）]\s*(.+)$',
        # Special keywords that often start sections
        r'^[\s]*(背景|BACKGROUND|交易|TRANSACTION|財務影響|FINANCIAL IMPACT|風險|RISK)[:：]?\s*$',
        r'^[\s]*(董事會|BOARD|建議|RECOMMENDATION|條款|TERMS|先決條件|CONDITIONS PRECEDENT)[:：]?\s*$',
    ]
    
    # Table indicators
    TABLE_INDICATORS = [
        r'[-─━]+\s*[-─━]+',  # Table borders
        r'\|.*\|.*\|',  # Pipe-separated tables
        r'^\s*[^\s]+\s+\d+[,，]\d+',  # Financial figures alignment
        r'人民幣|RMB|HK\$|港元|USD|美元',  # Currency indicators
    ]
    
    # Important financial terms that should not be split
    FINANCIAL_TERMS = [
        '每股', '市盈率', '資產淨值', '股息', '派息', '配股', '供股',
        'earnings per share', 'P/E ratio', 'NAV', 'dividend', 'rights issue',
        '收購', '出售', '交易對價', '代價', 'acquisition', 'disposal',
        '關連交易', 'connected transaction', '主要交易', 'major transaction'
    ]
    
    def __init__(
        self, 
        chunk_size: int = 1500,  # Larger for financial docs
        chunk_overlap: int = 400,  # More overlap for context
        preserve_tables: bool = True,
        extract_metadata: bool = True
    ):
        """Initialize financial chunker with enhanced parameters.
        
        Args:
            chunk_size: Target chunk size in tokens (default 1500 for longer context)
            chunk_overlap: Overlap between chunks (default 400)
            preserve_tables: Whether to avoid splitting tables
            extract_metadata: Whether to extract document metadata
        """
        super().__init__(chunk_size, chunk_overlap)
        self.preserve_tables = preserve_tables
        self.extract_metadata = extract_metadata
        self._section_headers: List[Tuple[int, str]] = []
        
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect if a line is a section header."""
        line = line.strip()
        if not line:
            return None
            
        for pattern in self.SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return line
        return None
    
    def _is_in_table(self, text: str, position: int, window: int = 200) -> bool:
        """Check if position is likely within a table."""
        if not self.preserve_tables:
            return False
            
        # Look around the position for table indicators
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]
        
        for indicator in self.TABLE_INDICATORS:
            if re.search(indicator, context):
                return True
        return False
    
    def _contains_financial_term(self, text: str, position: int, window: int = 50) -> bool:
        """Check if position is near important financial terms."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()
        
        for term in self.FINANCIAL_TERMS:
            if term.lower() in context:
                return True
        return False
    
    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from document header."""
        metadata = {}
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        # Stock code pattern
        stock_code_pattern = r'股份代號|Stock Code|股票代碼[：:]\s*(\d+)'
        # Company name pattern  
        company_pattern = r'公司名稱|Company Name[：:]\s*(.+)'
        # Announcement type
        announcement_pattern = r'公告|ANNOUNCEMENT|通告|NOTICE[：:]\s*(.+)'
        
        for line in lines:
            # Stock code
            match = re.search(stock_code_pattern, line, re.IGNORECASE)
            if match:
                metadata['stock_code'] = match.group(1).strip()
            
            # Company name
            match = re.search(company_pattern, line, re.IGNORECASE)
            if match:
                metadata['company_name'] = match.group(1).strip()
                
            # Announcement type
            if '盈利' in line or 'earnings' in line.lower():
                metadata['type'] = 'earnings'
            elif '收購' in line or 'acquisition' in line.lower():
                metadata['type'] = 'acquisition'
            elif '關連交易' in line or 'connected transaction' in line.lower():
                metadata['type'] = 'connected_transaction'
                
        return metadata
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing for financial documents."""
        # First apply base preprocessing
        text = super()._preprocess_text(text)
        
        # Additional preprocessing for financial docs
        # Preserve table structures by not normalizing table whitespace
        lines = text.split('\n')
        processed_lines = []
        in_table = False
        
        for line in lines:
            # Detect table start/end
            if any(re.search(pattern, line) for pattern in self.TABLE_INDICATORS):
                in_table = True
            elif in_table and line.strip() == '':
                in_table = False
            
            if in_table:
                # Preserve original spacing in tables
                processed_lines.append(line)
            else:
                # Normal processing for non-table text
                processed_lines.append(line.strip())
        
        # Extract section headers for structure preservation
        self._section_headers = []
        for i, line in enumerate(processed_lines):
            header = self._detect_section_header(line)
            if header:
                self._section_headers.append((i, header))
        
        return '\n'.join(processed_lines)
    
    def _find_best_split_point(self, text: str, target_pos: int) -> int:
        """Enhanced split point detection for financial documents."""
        # Increase search window for financial documents
        search_window = min(200, len(text) // 3)
        
        start_search = max(0, target_pos - search_window)
        end_search = min(len(text), target_pos + search_window)
        
        best_pos = target_pos
        best_score = 0
        
        # Section headers have highest priority
        for i in range(start_search, end_search):
            score = 0
            
            # Check if this is right before a section header
            remaining_text = text[i:]
            first_line = remaining_text.split('\n')[0] if '\n' in remaining_text else remaining_text
            if self._detect_section_header(first_line):
                score = 15  # Highest priority
            
            # Avoid splitting in tables
            elif self._is_in_table(text, i):
                score = -10  # Negative score to avoid
                
            # Avoid splitting near financial terms
            elif self._contains_financial_term(text, i):
                score = -5
                
            # Standard sentence endings (from parent class)
            elif i < len(text) and text[i] in ['。', '！', '？', '；']:
                score = 10
            elif i < len(text) - 1 and text[i:i + 2] in ['. ', '! ', '? ', '; ']:
                score = 10
            
            # Paragraph breaks
            elif i < len(text) - 1 and text[i:i + 2] == '\n\n':
                score = 12  # Higher than sentence for financial docs
                
            # Single line breaks (often used in lists)
            elif i < len(text) and text[i] == '\n':
                score = 8
                
            # Numbered list items
            elif i < len(text) - 2 and re.match(r'\n\d+[\.)]\s', text[i:i+4]):
                score = 9
                
            # Lower priority for commas
            elif i < len(text) and text[i] in ['，', ',']:
                score = 2
            
            # Prefer positions closer to target
            distance_penalty = abs(i - target_pos) / search_window
            final_score = score * (1 - distance_penalty * 0.5)  # Less penalty for financial docs
            
            if final_score > best_score:
                best_score = final_score
                best_pos = i + 1 if score > 0 else i
        
        return best_pos
    
    async def chunk(self, text: str, return_metadata: bool = False) -> List[str] | Tuple[List[str], Dict]:
        """Chunk financial document with enhanced structure preservation.
        
        Args:
            text: The document text to chunk
            return_metadata: Whether to return extracted metadata
            
        Returns:
            List of chunks, optionally with metadata dict
        """
        if not text:
            return ([], {}) if return_metadata else []
        
        # Extract metadata if requested
        metadata = self._extract_document_metadata(text) if self.extract_metadata else {}
        
        # Preprocess with structure detection
        text = self._preprocess_text(text)
        
        # Get base chunks
        chunks = await super().chunk(text)
        
        # Post-process chunks to add section context
        enhanced_chunks = []
        for chunk in chunks:
            # Find which section this chunk belongs to
            chunk_start = text.find(chunk)
            current_section = None
            
            for pos, header in reversed(self._section_headers):
                if pos <= chunk_start:
                    current_section = header
                    break
            
            # Add section context if not already present
            if current_section and not chunk.startswith(current_section):
                # Only add if it doesn't make chunk too large
                with_context = f"{current_section}\n\n{chunk}"
                encoded = self.encoder.encode(with_context)
                if len(encoded) <= self.chunk_size * 1.2:  # Allow 20% overflow
                    chunk = with_context
            
            enhanced_chunks.append(chunk)
        
        if return_metadata:
            return enhanced_chunks, metadata
        return enhanced_chunks


# Convenience instance
financial_chunker = FinancialChunker()