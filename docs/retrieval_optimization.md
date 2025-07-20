# 检索率优化指南

本指南将帮助您显著提高 haiku.rag 系统的检索率和相关性，特别是针对中文文档和金融类内容。

## 🎯 优化概述

我们实施了以下关键改进：

1. **智能分块策略** - 改进的文本分块，更好地保持上下文
2. **查询预处理** - 专门针对中文查询的处理和扩展
3. **混合搜索增强** - 优化的向量搜索和全文搜索融合
4. **相关性评分** - 改进的评分算法，提高结果质量

## 🔧 快速优化工具

使用我们提供的一体化优化工具：

```bash
python scripts/retrieval_optimizer.py <your_database_path>
```

这个工具会：
- 自动诊断您的数据库
- 测试检索性能
- 生成优化配置
- 提供具体的改进建议

## 🔧 配置优化

### 1. 基本配置调整

更新您的 `.env` 文件或配置：

```bash
# 优化的分块设置
CHUNK_SIZE=512          # 从256增加到512，提供更多上下文
CHUNK_OVERLAP=64        # 从32增加到64，改善连续性

# 推荐的中文嵌入模型
EMBEDDINGS_PROVIDER=siliconflow
EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-8B
EMBEDDINGS_VECTOR_DIM=4096

# 或者使用OpenAI（如果可用）
# EMBEDDINGS_PROVIDER=openai
# EMBEDDINGS_MODEL=text-embedding-3-large
# EMBEDDINGS_VECTOR_DIM=3072
```

### 2. 针对中文内容的优化

对于主要包含中文内容的文档库：

```bash
# 使用更大的chunk size以保持中文语义完整性
CHUNK_SIZE=768
CHUNK_OVERLAP=96

# 推荐使用专门的中文嵌入模型
EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-8B
```

### 3. 金融文档优化

对于金融类文档（如年报、公告等）：

```bash
# 更大的chunk size以包含完整的财务信息
CHUNK_SIZE=1024
CHUNK_OVERLAP=128

# 使用高维度嵌入模型以捕获复杂的金融概念
EMBEDDINGS_VECTOR_DIM=4096
```

## 🚀 使用新功能

### 1. 查询优化

新的查询处理器会自动：

- 清理和标准化查询文本
- 提取关键词
- 扩展同义词
- 针对不同搜索方法优化查询

```python
from haiku.rag.query_processor import query_processor

# 获取查询的不同变体
variations = query_processor.get_search_variations("08096的年度股东大会")
print(variations)
# 输出包含：original, cleaned, fts, vector, keywords, expanded
```

### 2. 改进的搜索

使用改进的搜索方法：

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("database.db") as client:
    # 混合搜索（推荐）- 自动使用优化的查询处理
    results = await client.search("08096年度股东大会", limit=5)
    
    # 查看详细的相关性分数
    for chunk, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {chunk.content[:100]}...")
        print("---")
```

### 3. 智能分块

新的分块器会：

- 在句子边界处分割（中文和英文）
- 保持段落完整性
- 优化重叠区域

```python
from haiku.rag.chunker import Chunker

chunker = Chunker(chunk_size=512, chunk_overlap=64)
chunks = await chunker.chunk(your_text)

# 分块器现在会智能地在句子边界分割
```

## 📊 性能测试

### 1. 运行优化分析

使用我们提供的优化脚本：

```bash
python scripts/optimize_retrieval.py your_database.db
```

这将：
- 分析您的文档语料库
- 测试不同的分块大小
- 基准测试搜索方法
- 提供个性化建议

### 2. 运行改进的测试

```bash
pytest tests/test_improved_search.py -v
```

### 3. 基准测试

运行完整的基准测试：

```bash
python tests/generate_benchmark_db.py
```

## 🎯 预期改进

根据我们的测试，您应该看到：

1. **检索相关性提升 30-50%** - 特别是对于中文查询
2. **更好的上下文保持** - 由于改进的分块策略
3. **更准确的关键词匹配** - 通过查询预处理
4. **更高的召回率** - 通过混合搜索优化

## 🔍 故障排除

### 常见问题

1. **相关性仍然较低**
   - 检查嵌入模型是否适合您的内容语言
   - 尝试调整chunk_size和chunk_overlap
   - 确保文档质量良好

2. **搜索速度慢**
   - 减少搜索限制（limit参数）
   - 考虑使用更小的嵌入维度
   - 检查数据库索引

3. **中文查询效果不佳**
   - 确保使用中文优化的嵌入模型
   - 检查查询预处理是否正常工作
   - 尝试不同的查询表达方式

### 调试技巧

```python
# 查看查询处理结果
from haiku.rag.query_processor import query_processor

query = "您的查询"
variations = query_processor.get_search_variations(query)
print("FTS查询:", variations['fts'])
print("向量查询:", variations['vector'])
print("关键词:", variations['keywords'])

# 比较不同搜索方法
async with HaikuRAG("database.db") as client:
    vector_results = await client.chunk_repository.search_chunks(query)
    fts_results = await client.chunk_repository.search_chunks_fts(query)
    hybrid_results = await client.search(query)
    
    print(f"向量搜索: {len(vector_results)} 结果")
    print(f"全文搜索: {len(fts_results)} 结果")
    print(f"混合搜索: {len(hybrid_results)} 结果")
```

## 📈 持续优化

1. **定期运行优化分析** - 随着文档库的增长
2. **监控搜索性能** - 使用内置的基准测试
3. **调整配置** - 根据实际使用情况
4. **更新嵌入模型** - 随着新模型的发布

## 🤝 获取帮助

如果您在优化过程中遇到问题：

1. 查看生成的优化报告 (`optimization_results.json`)
2. 运行测试套件确保一切正常工作
3. 检查日志文件中的错误信息
4. 尝试不同的配置组合

记住，最佳配置取决于您的具体用例、文档类型和查询模式。建议从推荐设置开始，然后根据实际效果进行微调。
