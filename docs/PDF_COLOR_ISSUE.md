# PDF 颜色空间错误解决方案

## 错误信息
```
Cannot set non-stroke color because 2 components are specified but only 1 (grayscale), 3 (rgb) and 4 (cmyk) are supported
```

## 问题原因
某些 PDF 文件使用了特殊的颜色空间：
- **双色调（Duotone）**：使用 2 个颜色组件
- **Lab 颜色空间**：某些配置可能使用 2 个组件
- **其他专业印刷颜色格式**

`markitdown` 库的 PDF 处理器只支持标准颜色空间：
- 1 个组件：灰度
- 3 个组件：RGB
- 4 个组件：CMYK

## 解决方案

### 1. 临时解决（已实施）
已更新 `src/haiku/rag/reader.py`，提供更友好的错误提示。

### 2. 转换 PDF 文件
使用以下方法之一转换 PDF 到标准格式：

#### 方法 A：使用 Adobe Acrobat
1. 打开 PDF 文件
2. 文件 → 另存为 → 优化的 PDF
3. 选择"标准"预设
4. 确保颜色空间设置为 RGB

#### 方法 B：使用 Ghostscript（命令行）
```bash
# 安装 Ghostscript
# Windows: https://www.ghostscript.com/download/gsdnld.html

# 转换 PDF 到 RGB
gswin64c -sDEVICE=pdfwrite -sColorConversionStrategy=RGB -dProcessColorModel=/DeviceRGB -dCompatibilityLevel=1.4 -o output.pdf input.pdf
```

#### 方法 C：使用在线工具
- https://www.ilovepdf.com/
- https://smallpdf.com/

### 3. 提取纯文本
如果只需要文本内容，可以：

```python
# 使用 PyPDF2 提取纯文本
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 然后手动添加到 haiku-rag
text = extract_text_from_pdf("problematic.pdf")
os.system(f'haiku-rag add "{text}"')
```

### 4. 使用其他工具预处理
```bash
# 使用 pdftotext（更稳定）
pdftotext input.pdf output.txt
haiku-rag add-src output.txt
```

## 预防措施

1. **检查 PDF 来源**：专业印刷或设计软件导出的 PDF 更容易有特殊颜色空间
2. **标准化工作流程**：在创建 PDF 时选择 RGB 或 sRGB 颜色空间
3. **批量处理**：对于大量 PDF，考虑编写脚本批量转换

## 未来改进建议

1. 在 `FileReader` 中添加降级处理，尝试只提取文本
2. 集成其他 PDF 处理库作为后备方案（如 pdfplumber）
3. 添加命令行参数跳过有问题的文件