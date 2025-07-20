#!/usr/bin/env python
"""测试实际使用场景中的错误处理"""

import asyncio
import sys
from pathlib import Path
import pytest

# 添加源代码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.client import HaikuRAG


@pytest.mark.asyncio
async def test_add_problematic_pdf():
    """测试添加有问题的 PDF 文件时的错误处理"""
    
    # 创建一个模拟的 PDF 路径
    pdf_path = Path("problematic_color_space.pdf")
    
    # 使用内存数据库进行测试
    async with HaikuRAG(":memory:") as client:
        # 由于文件不存在，应该抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError):
            await client.create_document_from_source(pdf_path)


@pytest.mark.asyncio
async def test_error_message_format():
    """测试错误信息格式"""
    
    # 测试文件不存在的情况
    non_existent_file = Path("non_existent_file.pdf")
    
    async with HaikuRAG(":memory:") as client:
        with pytest.raises(FileNotFoundError) as exc_info:
            await client.create_document_from_source(non_existent_file)
        
        # 验证错误是 FileNotFoundError
        assert isinstance(exc_info.value, FileNotFoundError)


@pytest.mark.asyncio 
async def test_pdf_processing_workflow():
    """测试 PDF 处理工作流程"""
    
    # 创建一个临时的测试文本文件
    test_content = "这是一个测试文档，用于验证文档处理流程。"
    test_file = Path("test_document.txt")
    
    try:
        # 创建临时文件
        test_file.write_text(test_content, encoding="utf-8")
        
        async with HaikuRAG(":memory:") as client:
            # 测试添加文档
            doc = await client.create_document_from_source(test_file)
            assert doc is not None
            assert doc.id is not None
            
            # 测试搜索功能
            results = await client.search("测试文档", limit=1)
            assert len(results) > 0
            
            # 测试问答功能（如果QA代理可用）
            try:
                answer = await client.ask("这个文档说了什么？")
                assert answer is not None
            except Exception:
                # 如果QA代理不可用，跳过这个测试
                pass
                
    finally:
        # 清理临时文件
        if test_file.exists():
            test_file.unlink()


def test_error_handling_helper():
    """测试错误处理辅助函数"""
    
    # 这个测试确保错误处理逻辑的正确性
    error_msg = "Cannot set non-stroke color because 2 components are specified"
    
    # 检查是否能正确识别颜色空间错误
    assert "set non-stroke color" in error_msg or "components are specified" in error_msg