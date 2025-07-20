# 动态切块策略文档
# Dynamic Chunking Strategy Documentation

## 概述 Overview

Haiku RAG 现在支持动态切块策略，特别针对金融文档进行了优化。动态切块策略可以根据文档内容和位置智能调整切块大小，在保持语义完整性的同时提供更灵活的切块控制。

## 配置参数 Configuration Parameters

### 基础参数

```bash
# 启用金融文档切块器
USE_FINANCIAL_CHUNKER=true

# 动态切块参数
FINANCIAL_CHUNK_SIZE=500            # 目标切块大小 (tokens)
FINANCIAL_CHUNK_OVERLAP=100         # 重叠大小 (tokens)
FINANCIAL_MIN_CHUNK_SIZE=300        # 最小切块大小 (tokens)
FINANCIAL_MAX_CHUNK_SIZE=500        # 最大切块大小 (tokens)
FINANCIAL_CHUNK_SIZE_VARIANCE=100   # 切块大小浮动范围 (tokens)

# 功能开关
PRESERVE_TABLES=true                # 保护表格完整性
EXTRACT_METADATA=true               # 提取文档元数据
```

### 参数说明

- **FINANCIAL_CHUNK_SIZE**: 目标切块大小，系统会尽量接近这个值
- **FINANCIAL_MIN_CHUNK_SIZE**: 最小允许的切块大小
- **FINANCIAL_MAX_CHUNK_SIZE**: 最大允许的切块大小
- **FINANCIAL_CHUNK_SIZE_VARIANCE**: 切块大小的浮动范围
- **FINANCIAL_CHUNK_OVERLAP**: 相邻切块之间的重叠token数

## 动态切块逻辑 Dynamic Chunking Logic

### 1. 位置自适应

系统会根据文档中的位置动态调整切块大小：

- **文档开头 (0-10%)**: 使用较小的切块，便于快速定位关键信息
- **文档中间 (10-90%)**: 使用较大的切块，保持更多上下文
- **文档结尾 (90-100%)**: 使用较小的切块，处理总结性内容

### 2. 内容感知分割

- **句子边界优先**: 优先在句子结尾处分割
- **段落完整性**: 尽量保持段落的完整性
- **表格保护**: 避免在表格中间分割
- **金融术语保护**: 避免分割重要的金融术语

### 3. 智能重叠

- 确保相邻切块之间有适当的重叠
- 重叠区域包含关键的上下文信息
- 避免重要信息在切块边界丢失

## 使用示例 Usage Examples

### Python API

```python
from haiku.rag.domains.financial.chunker import FinancialChunker

# 创建动态切块器
chunker = FinancialChunker(
    chunk_size=500,           # 目标大小
    chunk_overlap=100,        # 重叠
    min_chunk_size=300,       # 最小大小
    max_chunk_size=500,       # 最大大小
    chunk_size_variance=100,  # 浮动范围
    preserve_tables=True,
    extract_metadata=True
)

# 执行切块
chunks = await chunker.chunk(document_text)

# 分析切块结果
for i, chunk in enumerate(chunks):
    tokens = chunker.encoder.encode(chunk)
    print(f"块 {i+1}: {len(tokens)} tokens")
```

### 配置验证

```python
# 验证切块质量
token_counts = [len(chunker.encoder.encode(chunk)) for chunk in chunks]
in_range_count = sum(1 for size in token_counts 
                    if chunker.min_chunk_size <= size <= chunker.max_chunk_size)
print(f"符合范围的块: {in_range_count}/{len(chunks)}")
```

## 性能优化建议 Performance Optimization

### 1. 不同文档类型的推荐配置

#### 港交所公告
```bash
FINANCIAL_CHUNK_SIZE=500
FINANCIAL_MIN_CHUNK_SIZE=300
FINANCIAL_MAX_CHUNK_SIZE=500
FINANCIAL_CHUNK_OVERLAP=100
FINANCIAL_CHUNK_SIZE_VARIANCE=100
```

#### 年报文档
```bash
FINANCIAL_CHUNK_SIZE=600
FINANCIAL_MIN_CHUNK_SIZE=400
FINANCIAL_MAX_CHUNK_SIZE=800
FINANCIAL_CHUNK_OVERLAP=150
FINANCIAL_CHUNK_SIZE_VARIANCE=200
```

#### 简短公告
```bash
FINANCIAL_CHUNK_SIZE=300
FINANCIAL_MIN_CHUNK_SIZE=200
FINANCIAL_MAX_CHUNK_SIZE=400
FINANCIAL_CHUNK_OVERLAP=50
FINANCIAL_CHUNK_SIZE_VARIANCE=100
```

### 2. 质量指标

- **切块大小分布**: 大部分切块应在目标范围内
- **重叠率**: 通常在20-40%之间
- **语义完整性**: 重要信息不应被分割

## 测试和验证 Testing and Validation

### 运行测试脚本

```bash
python test_dynamic_chunking.py
```

### 预期输出

```
=== 测试动态切块策略 ===

切块器配置:
  目标大小: 500 tokens
  最小大小: 300 tokens
  最大大小: 500 tokens
  浮动范围: 100 tokens
  重叠大小: 100 tokens

切块结果:
  总块数: 3

块 1:
  Token数量: 348
  字符数量: 431
  是否在范围内: True

=== 切块质量分析 ===
原文Token数: 771
切块总Token数: 1036
重叠率: 34.4%
平均块大小: 345.3 tokens
符合范围的块: 2/3 (66.7%)
```

## 故障排除 Troubleshooting

### 常见问题

1. **切块过大**: 减小 `FINANCIAL_MAX_CHUNK_SIZE`
2. **切块过小**: 增大 `FINANCIAL_MIN_CHUNK_SIZE`
3. **重叠过多**: 减小 `FINANCIAL_CHUNK_OVERLAP`
4. **语义分割**: 检查 `PRESERVE_TABLES` 设置

### 调试技巧

- 使用测试脚本验证配置
- 检查切块大小分布
- 分析重叠率是否合理
- 确认重要信息未被分割

## 更新日志 Changelog

- **v1.0**: 实现基础动态切块功能
- **v1.1**: 添加位置自适应逻辑
- **v1.2**: 优化金融术语保护
- **v1.3**: 改进表格检测和保护
