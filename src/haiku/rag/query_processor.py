import re
from typing import List, Set


class QueryProcessor:
    """Query processor for improving search quality, especially for Chinese text."""
    
    def __init__(self):
        # Common stop words for Chinese and English
        self.chinese_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还是', '为了', '还有', '可以', '这个', '那个',
            '什么', '怎么', '为什么', '哪里', '哪个', '怎样', '如何', '多少', '几个'
        }
        
        self.english_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Financial and business terms that should be preserved
        self.important_terms = {
            '股东大会', '年度报告', '财务报表', '董事会', '监事会', '审计', '合并', '收购',
            '投资', '利润', '营收', '资产', '负债', '现金流', '股价', '分红', '配股',
            'AGM', 'annual meeting', 'financial report', 'board', 'audit', 'merger',
            'acquisition', 'investment', 'profit', 'revenue', 'assets', 'liabilities',
            'cash flow', 'stock price', 'dividend'
        }

    def clean_query(self, query: str) -> str:
        """Clean and normalize the query text."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters but keep Chinese characters, letters, numbers
        query = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', query)
        
        # Normalize whitespace again
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query

    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from the query with aggressive Chinese processing."""
        query = self.clean_query(query)

        keywords = []

        # For Chinese text, use character-level and word-level extraction
        if any('\u4e00' <= char <= '\u9fff' for char in query):
            # Chinese text processing

            # Extract numbers (like stock codes)
            import re
            numbers = re.findall(r'\d+', query)
            keywords.extend(numbers)

            # Extract Chinese words (2+ characters)
            chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', query)
            keywords.extend(chinese_words)

            # Extract individual meaningful Chinese characters
            meaningful_chars = []
            for char in query:
                if '\u4e00' <= char <= '\u9fff':
                    meaningful_chars.append(char)

            # Group consecutive Chinese characters
            if meaningful_chars:
                current_word = ""
                for char in query:
                    if '\u4e00' <= char <= '\u9fff':
                        current_word += char
                    else:
                        if len(current_word) >= 1:
                            keywords.append(current_word)
                        current_word = ""
                if current_word:
                    keywords.append(current_word)

        # Standard word splitting
        words = query.split()
        for word in words:
            word_lower = word.lower()

            # Skip if it's a stop word (but preserve important terms)
            if (word_lower in self.chinese_stop_words or
                word_lower in self.english_stop_words) and \
               word_lower not in self.important_terms:
                continue

            # Skip very short words unless they're important
            if len(word) < 2 and word_lower not in self.important_terms:
                continue

            keywords.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        # Synonym mapping for common financial terms
        synonyms = {
            '股东大会': ['股东会议', '股东周年大会', 'AGM', 'annual general meeting'],
            '年度报告': ['年报', 'annual report', '年度财务报告'],
            '财务报表': ['财报', 'financial statement', '财务报告'],
            '董事会': ['board of directors', '董事局'],
            '审计': ['audit', '审计报告', 'auditing'],
            '投资': ['investment', '投资项目', '投资计划'],
            '收购': ['acquisition', '并购', 'merger'],
            '利润': ['profit', '盈利', '净利润', 'net profit'],
            '营收': ['revenue', '收入', '营业收入'],
            '股价': ['stock price', '股票价格', 'share price'],
            '分红': ['dividend', '股息', '派息']
        }
        
        expanded_queries = [query]
        keywords = self.extract_keywords(query)
        
        # Add synonym variations
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in synonyms:
                for synonym in synonyms[keyword_lower]:
                    # Create new query by replacing the keyword with synonym
                    new_query = query.replace(keyword, synonym)
                    if new_query != query:
                        expanded_queries.append(new_query)
        
        return expanded_queries

    def process_for_fts(self, query: str) -> str:
        """Process query specifically for FTS5 full-text search with aggressive matching."""
        keywords = self.extract_keywords(query)

        if not keywords:
            return query

        # For FTS5, use multiple strategies for better recall
        fts_terms = []

        # Strategy 1: Individual keywords with OR
        meaningful_keywords = [k for k in keywords if len(k) >= 1]  # Include single chars for Chinese
        if meaningful_keywords:
            fts_terms.extend(meaningful_keywords)

        # Strategy 2: Exact phrase matching
        if len(keywords) > 1:
            phrase = ' '.join(keywords)
            fts_terms.append(f'"{phrase}"')

        # Strategy 3: Partial matching for Chinese characters
        chinese_chars = [k for k in keywords if len(k) == 1 and '\u4e00' <= k <= '\u9fff']
        if chinese_chars:
            # Add individual Chinese characters
            fts_terms.extend(chinese_chars)

        # Strategy 4: Number matching (for stock codes)
        numbers = [k for k in keywords if k.isdigit()]
        if numbers:
            fts_terms.extend(numbers)

        # Remove duplicates
        unique_terms = list(dict.fromkeys(fts_terms))

        if unique_terms:
            # Use OR for broader matching
            return ' OR '.join(unique_terms)
        else:
            return query

    def process_for_vector(self, query: str) -> str:
        """Process query for vector similarity search."""
        # For vector search, we want to preserve context and meaning
        # Clean the query but keep it as natural language
        cleaned = self.clean_query(query)
        
        # Add context hints for better embedding
        if any(term in cleaned.lower() for term in ['股东大会', 'agm', 'annual meeting']):
            cleaned += " 股东大会 年度会议"
        elif any(term in cleaned.lower() for term in ['财务', 'financial', '报告', 'report']):
            cleaned += " 财务报告 年度报告"
        elif any(term in cleaned.lower() for term in ['投票', 'voting', '决议', 'resolution']):
            cleaned += " 投票结果 股东决议"
            
        return cleaned

    def get_search_variations(self, query: str) -> dict:
        """Get different variations of the query for different search methods."""
        return {
            'original': query,
            'cleaned': self.clean_query(query),
            'fts': self.process_for_fts(query),
            'vector': self.process_for_vector(query),
            'keywords': self.extract_keywords(query),
            'expanded': self.expand_query(query)
        }


# Global instance
query_processor = QueryProcessor()
