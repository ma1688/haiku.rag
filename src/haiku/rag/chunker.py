import re
from typing import ClassVar

import tiktoken

from haiku.rag.config import Config


class Chunker:
    """将文本切分为更小的部分以便于嵌入和检索的类。

    参数:
        chunk_size: 切块的最大大小(以 tokens 为单位)。
        chunk_overlap: 切块之间的重叠数量(以 tokens 为单位)。
    """

    # 定义字符编码器，使用 tiktoken 库中的特定模型
    encoder: ClassVar[tiktoken.Encoding] = tiktoken.encoding_for_model("gpt-4o")

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size  # 设置切块大小
        self.chunk_overlap = chunk_overlap  # 设置切块重叠量

    def _preprocess_text(self, text: str) -> str:
        """对文本进行预处理，以便更好地进行切块，特别是针对中文文档。

        参数:
            text: 需要进行切块的文本。

        返回:
            经过处理的文本字符串。
        """
        # 归一化空白字符
        text = re.sub(r'\s+', ' ', text)

        # 为中文文本添加句子边界
        # 中文句子结尾符号
        text = re.sub(r'([。！？；])', r'\1\n', text)

        # 英文句子结尾符号
        text = re.sub(r'([.!?;])\s+', r'\1\n', text)

        # 合并多段空行
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()  # 返回去除前后空白后的文本

    def _find_best_split_point(self, text: str, target_pos: int) -> int:
        """寻找文本的最佳分割位置，优先靠近句子边界。

        参数:
            text: 尚未切块的文本。
            target_pos: 期望的分割位置。

        返回:
            最佳分割位置。
        """
        search_window = min(100, len(text) // 4)  # 定义搜索窗口大小

        start_search = max(0, target_pos - search_window)
        end_search = min(len(text), target_pos + search_window)

        # 中文句子结尾符号
        chinese_endings = ['。', '！', '？', '；']
        # 英文句子结尾符号
        english_endings = ['. ', '! ', '? ', '; ']

        best_pos = target_pos
        best_score = 0

        # 在搜索窗口内查找最佳分割位置
        for i in range(start_search, end_search):
            score = 0

            # 检查是否是中文句子结尾
            if i < len(text) and text[i] in chinese_endings:
                score = 10
            # 检查是否是英文句子结尾
            elif i < len(text) - 1 and text[i:i + 2] in english_endings:
                score = 10
            # 检查是否是段落换行
            elif i < len(text) - 1 and text[i:i + 2] == '\n\n':
                score = 8
            # 检查是否是单行换行
            elif i < len(text) and text[i] == '\n':
                score = 5
            # 检查逗号（优先级较低）
            elif i < len(text) and text[i] in ['，', ',']:
                score = 2

            # 根据距离调整分数
            distance_penalty = abs(i - target_pos) / search_window
            final_score = score * (1 - distance_penalty)

            # 更新最佳分割位置
            if final_score > best_score:
                best_score = final_score
                best_pos = i + 1 if score > 0 else i

        return best_pos

    async def chunk(self, text: str) -> list[str]:
        """根据 token 边界进行智能切块。

        参数:
            text: 需要切块的文本。

        返回:
            以 token 为边界并带有重叠的文本切块列表。
        """
        if not text:
            return []  # 若文本为空，返回空列表

        # 对文本进行预处理
        text = self._preprocess_text(text)

        # 编码文本为 tokens 列表
        encoded_tokens = self.encoder.encode(text, disallowed_special=())

        if self.chunk_size > len(encoded_tokens):
            return [text]  # 若切块大小大于文本长度，返回整个文本

        chunks = []  # 用于存储切块的列表
        i = 0  # 当前在 tokens 列表中的起始位置

        while i < len(encoded_tokens):
            start_i = i
            end_i = min(i + self.chunk_size, len(encoded_tokens))

            # 尝试找到更好的分割点
            if end_i < len(encoded_tokens):
                # 解码当前 tokens 为临时文本
                chunk_tokens = encoded_tokens[start_i:end_i]
                temp_text = self.encoder.decode(chunk_tokens)

                # 找到更好的分割点
                better_end = self._find_best_split_point(temp_text, len(temp_text))

                # 检查是否需要更新结束位置
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

        return chunks  # 返回切块列表


# 实例化 Chunker
chunker = Chunker()
