#!/usr/bin/env python
"""测试 PDF 颜色空间错误处理"""

import sys
from pathlib import Path
import pytest

# 添加源代码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.reader import FileReader


def test_pdf_color_space_error_handling():
    """测试改进后的错误处理"""
    
    # 模拟一个会产生颜色空间错误的异常
    class MockMarkItDown:
        def convert(self, path):
            # 模拟 markitdown 抛出的颜色空间错误
            raise Exception("Cannot set non-stroke color because 2 components are specified but only 1 (grayscale), 3 (rgb) and 4 (cmyk) are supported")
    
    # 保存原始的 MarkItDown
    import haiku.rag.reader
    original_markitdown = haiku.rag.reader.MarkItDown
    
    try:
        # 替换为 mock
        haiku.rag.reader.MarkItDown = MockMarkItDown
        
        # 测试 PDF 文件
        test_pdf = Path("test.pdf")
        
        # 应该抛出 ValueError
        with pytest.raises(ValueError) as exc_info:
            FileReader.parse_file(test_pdf)
        
        # 验证错误信息
        error_msg = str(exc_info.value)
        assert "unsupported color format" in error_msg
        assert "duotone or Lab colors" in error_msg
        assert "Try converting" in error_msg
            
    finally:
        # 恢复原始的 MarkItDown
        haiku.rag.reader.MarkItDown = original_markitdown


def test_pdf_color_space_error_message_content():
    """测试错误信息内容是否符合预期"""
    
    class MockMarkItDown:
        def convert(self, path):
            raise Exception("Cannot set non-stroke color because 2 components are specified")
    
    import haiku.rag.reader
    original_markitdown = haiku.rag.reader.MarkItDown
    
    try:
        haiku.rag.reader.MarkItDown = MockMarkItDown
        test_pdf = Path("test.pdf")
        
        with pytest.raises(ValueError) as exc_info:
            FileReader.parse_file(test_pdf)
        
        error_msg = str(exc_info.value)
        
        # 检查错误信息的友好性
        assert "unsupported color format" in error_msg
        assert "duotone or Lab colors" in error_msg
        assert "Try converting" in error_msg
        assert "extracting" in error_msg
            
    finally:
        haiku.rag.reader.MarkItDown = original_markitdown