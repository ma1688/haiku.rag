# Windows 环境下 haiku-rag chat 命令修复指南

## 问题描述
在 Windows 环境下运行 `haiku-rag chat` 时出现错误：
- DeprecationWarning: event_loop = asyncio.get_event_loop()
- Error: No such command 'chat'

## 问题原因
您正在使用通过 pip 安装的旧版本 haiku.rag，而不是本地开发版本。

## 解决方案

### 方案 1：使用开发模式安装（推荐）

在项目根目录下运行：

```powershell
# 卸载已安装的版本
pip uninstall haiku.rag -y

# 以开发模式安装本地版本
pip install -e .
```

这将安装本地版本并使用我们修复后的代码。

### 方案 2：直接运行本地版本

```powershell
# 使用 Python 模块方式运行
python -m haiku.rag.cli chat

# 或者设置 PYTHONPATH
$env:PYTHONPATH = ".\src"
python src/haiku/rag/cli.py chat
```

### 方案 3：更新已安装的包文件（临时方案）

找到并编辑已安装的文件：
```
D:\py_pro\haiku.rag\.venv\Lib\site-packages\haiku\rag\cli.py
```

在第 28-32 行，将：
```python
@cli.callback()
def main():
    """haiku.rag CLI - SQLite-based RAG system"""
    # Run version check before any command
    asyncio.run(check_version())
```

修改为：
```python
@cli.callback()
def main():
    """haiku.rag CLI - SQLite-based RAG system"""
    # Run version check before any command
    try:
        asyncio.run(check_version())
    except Exception:
        # Skip version check if it fails
        pass
```

并删除第 15 行的 `event_loop = asyncio.get_event_loop()`（如果存在）。

## 验证修复

修复后运行：
```powershell
haiku-rag chat --help
```

应该能看到 chat 命令的帮助信息。

## 其他注意事项

1. **ffmpeg 警告**：这是 pydub 的警告，如果不处理音频文件可以忽略。

2. **数据库路径**：Windows 下默认数据库位置在 `%APPDATA%\haiku.rag\`

3. **配置嵌入服务**：确保已配置好嵌入服务（Ollama/OpenAI/SiliconFlow）