"""
简化的交互式问答系统

提供基本的交互式问答功能：
- 上下文感知的对话管理
- 简洁的控制台界面
- 基本的搜索功能
- 会话历史管理
"""
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.logging import get_logger
from haiku.rag.monitor import FileWatcher
from haiku.rag.qa import get_qa_agent
from haiku.rag.qa.base import QuestionAnswerAgentBase

# 常量和配置
logger = get_logger()


@dataclass
class SessionConfig:
    """会话配置类"""
    max_history: int = 10  # 最大历史记录数
    max_context_length: int = 2000  # 最大上下文长度
    search_limit: int = 5  # 搜索结果限制
    context_window: int = 3  # 上下文窗口大小


@dataclass
class ConversationExchange:
    """单次对话交换记录"""
    timestamp: datetime  # 时间戳
    question: str  # 问题
    answer: str  # 答案
    search_results: List[Tuple[Any, float]]  # 搜索结果
    response_time: float  # 响应时间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于序列化"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "question": self.question,
            "answer": self.answer,
            "search_results": [(str(chunk), score) for chunk, score in self.search_results],
            "response_time": self.response_time
        }


class ConversationHistory:
    """对话历史管理类"""

    def __init__(self, config: SessionConfig, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id or self._generate_session_id()
        self.history: List[ConversationExchange] = []
        self.session_start = datetime.now()

    def _generate_session_id(self) -> str:
        """生成唯一的会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"

    def add_exchange(self, question: str, answer: str, search_results: Optional[List] = None,
                     response_time: float = 0.0):
        """添加问答交换记录到历史"""
        exchange = ConversationExchange(
            timestamp=datetime.now(),
            question=question,
            answer=answer,
            search_results=search_results or [],
            response_time=response_time
        )

        self.history.append(exchange)

        # 保持最近的交换记录
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]

        logger.debug(f"已添加交换记录到历史。总数: {len(self.history)}")

    def get_context_summary(self, max_length: Optional[int] = None) -> str:
        """从最近的对话历史生成上下文摘要"""
        if not self.history:
            return ""

        max_length = max_length or self.config.max_context_length
        context_parts = []
        current_length = 0

        # 使用最近的N次交换作为上下文
        recent_exchanges = self.history[-self.config.context_window:]

        for exchange in recent_exchanges:
            question_part = f"问: {exchange.question}"
            answer_preview = exchange.answer[:200]  # 简化预览长度
            if len(exchange.answer) > 200:
                answer_preview += "..."
            answer_part = f"答: {answer_preview}"

            exchange_text = f"{question_part}\n{answer_part}\n"

            if current_length + len(exchange_text) > max_length:
                break

            context_parts.append(exchange_text)
            current_length += len(exchange_text)

        return "\n".join(context_parts)

    def clear(self):
        """清空对话历史"""
        self.history.clear()
        self.session_start = datetime.now()
        logger.info("对话历史已清空")

    def save_to_file(self, file_path: Optional[Path] = None) -> Path:
        """保存对话历史到文件"""
        if not file_path:
            file_path = Path.home() / ".haiku_rag" / "sessions" / f"{self.session_id}.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "history": [exchange.to_dict() for exchange in self.history]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        logger.info(f"会话已保存到 {file_path}")
        return file_path


class ContextAwareQAAgent(QuestionAnswerAgentBase):
    """带上下文感知的问答代理"""

    def __init__(self, client: HaikuRAG, model: str = "", config: Optional[SessionConfig] = None):
        super().__init__(client, model)
        self.config = config or SessionConfig()
        self.base_agent = get_qa_agent(client, model)
        self.conversation_history = ConversationHistory(self.config)

    async def answer_with_context(self, question: str) -> Tuple[str, List]:
        """使用上下文回答问题"""
        start_time = time.time()

        try:
            # 输入验证
            if not question or not question.strip():
                raise ValueError("问题不能为空")

            question = question.strip()

            # 获取搜索结果
            search_results = await self._client.search(question, limit=self.config.search_limit)

            # 获取对话上下文
            context = self.conversation_history.get_context_summary(self.config.max_context_length)

            # 创建增强的问题
            enhanced_question = self._create_enhanced_question(question, context, search_results)

            # 从基础代理获取答案
            try:
                answer = await self.base_agent.answer(enhanced_question)
            except Exception as e:
                logger.error(f"从基础代理获取答案时出错: {e}")
                # 回退到不带上下文的简单问题
                answer = await self.base_agent.answer(question)
                logger.info("使用了不带上下文的回退答案")

            # 计算响应时间
            response_time = time.time() - start_time

            # 添加到对话历史
            self.conversation_history.add_exchange(
                question=question,
                answer=answer,
                search_results=search_results,
                response_time=response_time
            )

            logger.info(f"问题在 {response_time:.2f}s 内得到回答")
            return answer, search_results

        except Exception as e:
            logger.error(f"answer_with_context 中出错: {e}")
            response_time = time.time() - start_time

            # 添加失败的交换记录用于调试
            self.conversation_history.add_exchange(
                question=question,
                answer=f"错误: {str(e)}",
                search_results=[],
                response_time=response_time
            )

            raise

    def _create_enhanced_question(self, question: str, context: str, search_results: List) -> str:
        """创建带有上下文和搜索结果的增强问题"""
        if not context:
            return question

        # 创建上下文感知的提示
        enhanced_parts = []

        if context and len(context) < self.config.max_context_length:
            enhanced_parts.append(f"之前的对话上下文:\n{context}")

        # 添加相关的搜索上下文
        if search_results:
            search_context = self._create_search_context(search_results[:3])
            if search_context:
                enhanced_parts.append(f"知识库中的相关信息:\n{search_context}")

        enhanced_parts.append(f"当前问题: {question}")
        enhanced_parts.append("请回答当前问题，如果适用的话，请考虑对话上下文和相关信息。请简洁但全面地回答。")

        return "\n\n".join(enhanced_parts)

    def _create_search_context(self, search_results: List) -> str:
        """从搜索结果创建上下文"""
        if not search_results:
            return ""

        context_parts = []
        for i, (chunk, score) in enumerate(search_results, 1):
            if hasattr(chunk, 'content') and chunk.content:
                preview = chunk.content[:200].replace('\n', ' ').strip()
                if len(chunk.content) > 200:
                    preview += "..."
                context_parts.append(f"{i}. {preview}")

        return "\n".join(context_parts)

    async def answer(self, question: str) -> str:
        """标准答案方法，用于兼容性"""
        answer, _ = await self.answer_with_context(question)
        return answer


class InteractiveQASession:
    """简化的交互式问答会话"""

    def __init__(self, db_path: str, model: str = "", enable_monitoring: bool = True,
                 config: Optional[SessionConfig] = None):
        self.db_path = db_path
        self.model = model
        self.enable_monitoring = enable_monitoring
        self.config = config or SessionConfig()

        # 控制台组件
        self.console = Console()

        # 核心组件
        self.client: Optional[HaikuRAG] = None
        self.qa_agent: Optional[ContextAwareQAAgent] = None
        self.monitor: Optional[FileWatcher] = None
        self.monitor_task: Optional[asyncio.Task] = None

        # 会话管理
        self._session_start_time = time.time()
        self._is_running = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        try:
            self.console.print("🚀 正在初始化会话...")

            # 初始化客户端
            self.console.print("📁 连接到数据库...")
            self.client = HaikuRAG(self.db_path)
            await self.client.__aenter__()

            # 初始化问答代理
            self.console.print("🤖 设置问答代理...")
            self.qa_agent = ContextAwareQAAgent(self.client, self.model, self.config)

            # 设置文件监控
            await self._setup_file_monitoring()

            self._is_running = True
            logger.info(f"交互式问答会话初始化成功。会话ID: {self.qa_agent.conversation_history.session_id}")

            return self

        except Exception as e:
            logger.error(f"初始化会话失败: {e}")
            await self._cleanup()
            raise

    async def _setup_file_monitoring(self):
        """设置文件监控"""
        if not self.enable_monitoring:
            return

        if Config.MONITOR_DIRECTORIES:
            try:
                self.monitor = FileWatcher(paths=Config.MONITOR_DIRECTORIES, client=self.client)
                self.monitor_task = asyncio.create_task(self.monitor.observe())

                self.console.print(Panel(
                    f"📁 文件监控已启用: {', '.join(str(p) for p in Config.MONITOR_DIRECTORIES)}",
                    title="[green]🔍 文件监控激活", border_style="green"
                ))
                logger.info(f"文件监控已启动，监控 {len(Config.MONITOR_DIRECTORIES)} 个目录")

            except Exception as e:
                logger.error(f"启动文件监控失败: {e}")
                self.console.print(Panel(
                    f"❌ 文件监控启动失败: {str(e)}\n📝 会话将在没有文件监控的情况下继续。",
                    title="[red]🔍 文件监控错误", border_style="red"
                ))
        else:
            self.console.print(Panel(
                "⚠️ 文件监控已启用但未配置 MONITOR_DIRECTORIES。\n"
                "在 .env 文件中设置 MONITOR_DIRECTORIES 以启用自动文件监控。\n"
                "📖 示例: MONITOR_DIRECTORIES=/path/to/docs,/path/to/files",
                title="[yellow]📁 文件监控配置", border_style="yellow"
            ))

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self._cleanup()
        return False

    async def _cleanup(self):
        """清理所有资源"""
        self._is_running = False

        try:
            # 保存最终会话数据
            if self.qa_agent and self.qa_agent.conversation_history.history:
                try:
                    session_file = self.qa_agent.conversation_history.save_to_file()
                    self.console.print(Panel(
                        f"💾 会话已保存到: {session_file}\n"
                        f"📊 总交换次数: {len(self.qa_agent.conversation_history.history)}",
                        title="[green]💾 会话已保存", border_style="green"
                    ))
                except Exception as e:
                    logger.error(f"保存会话失败: {e}")

            # 停止文件监控
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            # 停止文件监控
            if self.monitor_task:
                self.console.print(Panel(
                    "📁 文件监控已停止。\n🔄 所有监控的文件都已处理。",
                    title="[blue]🔍 文件监控", border_style="blue"
                ))

            # 关闭客户端连接
            if self.client:
                await self.client.__aexit__(None, None, None)

            logger.info("会话清理成功完成")

        except Exception as e:
            logger.error(f"清理过程中出错: {e}")

    def _display_welcome(self):
        """显示欢迎信息"""
        welcome_text = f"""
🚀 欢迎使用交互式问答系统！

📋 会话信息:
   • 系统: HKEX ANNOUNCEMENT RAG 交互式问答
   • 版本: v0.2.0 (优化版)
   • 开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
   • 会话ID: {self.qa_agent.conversation_history.session_id if self.qa_agent else "初始化中..."}
   • 模型: {self.model or "默认"}
   • 数据库: {Path(self.db_path).name}

⚡ 可用命令:
   • /help - 显示帮助信息
   • /history - 显示对话历史
   • /clear - 清空对话历史
   • /search <查询> - 直接搜索文档
   • /refresh - 刷新监控目录
   • /save - 保存当前会话
   • /quit 或 /exit - 退出

💡 使用提示:
   • 我会记住我们的对话！可以自然地提出后续问题
   • 详细的问题会得到更好、更准确的答案
   • 我会显示回答所依据的文档来源

✨ 准备好探索您的知识库了！问我任何问题或使用 /help 获取指导...
"""
        self.console.print(Panel(welcome_text, title="[bold bright_blue]🎉 欢迎", border_style="bright_blue"))

    def _display_question(self, question: str):
        """显示用户问题"""
        question_count = len(self.qa_agent.conversation_history.history) + 1 if self.qa_agent else 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.console.print(f"\n[{timestamp}] 问题 #{question_count}")
        self.console.print(Panel(question, title="[bold bright_blue]👤 您的问题", border_style="bright_blue"))

    def _display_answer(self, answer: str, search_results: List = None, response_time: float = 0.0):
        """显示AI答案"""
        # 显示响应时间
        if response_time > 0:
            self.console.print(f"⏱️ 响应时间: {response_time:.2f}s")

        # 显示答案
        try:
            answer_panel = Panel(Markdown(answer), title="[bold bright_green]🤖 AI助手", border_style="bright_green")
        except Exception:
            # 如果markdown解析失败，回退到纯文本
            answer_panel = Panel(answer, title="[bold bright_green]🤖 AI助手", border_style="bright_green")

        self.console.print(answer_panel)

        # 显示搜索来源
        if search_results:
            self._display_search_sources(search_results)

    def _display_search_sources(self, search_results: List):
        """显示搜索来源"""
        if not search_results:
            return

        self.console.print("\n📚 知识来源:")
        for i, (chunk, score) in enumerate(search_results[:3], 1):
            # 格式化文档名称
            doc_name = "未知文档"
            if hasattr(chunk, 'document_uri') and chunk.document_uri:
                if chunk.document_uri.startswith('file://'):
                    doc_name = Path(chunk.document_uri).name
                else:
                    doc_name = chunk.document_uri

            # 格式化内容预览
            preview = ""
            if hasattr(chunk, 'content') and chunk.content:
                preview = chunk.content[:100].replace('\n', ' ').strip()
                if len(chunk.content) > 100:
                    preview += "..."

            # 相关性评分
            relevance = f"{score:.1%}"

            self.console.print(f"  {i}. [{relevance}] {doc_name}")
            if preview:
                self.console.print(f"     {preview}")
            self.console.print()

    async def _handle_search_command(self, query: str):
        """处理直接搜索命令"""
        if not query.strip():
            self.console.print("[red]❌ 请在 /search 命令后提供搜索查询。\n💡 示例: /search python 教程[/red]")
            return

        start_time = time.time()
        self.console.print("🔍 正在搜索知识库...")

        try:
            results = await self.client.search(query, limit=self.config.search_limit)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            self.console.print(f"[red]❌ 搜索失败: {str(e)}\n🔧 请重试或检查数据库连接。[/red]")
            return

        search_time = time.time() - start_time

        if not results:
            self.console.print(Panel(
                "🚫 未找到匹配的文档。\n\n"
                "💡 尝试这些建议:\n"
                "   • 使用不同的关键词\n"
                "   • 检查拼写\n"
                "   • 使用更广泛的搜索词\n"
                "   • 尝试同义词或相关词",
                title="[yellow]📭 未找到结果", border_style="yellow"
            ))
            return

        # 显示搜索结果
        self._display_search_results(results, query, search_time)

    def _display_search_results(self, results: List, query: str, search_time: float):
        """显示搜索结果"""
        self.console.print(f"\n🔍 搜索结果: '{query}' (找到 {len(results)} 个结果，用时 {search_time:.2f}s)")

        for i, (chunk, score) in enumerate(results, 1):
            # 格式化文档名称
            doc_name = "未知文档"
            if hasattr(chunk, 'document_uri') and chunk.document_uri:
                if chunk.document_uri.startswith('file://'):
                    doc_name = Path(chunk.document_uri).name
                else:
                    doc_name = chunk.document_uri

            # 格式化内容预览
            preview = ""
            if hasattr(chunk, 'content') and chunk.content:
                preview = chunk.content[:200].replace('\n', ' ').strip()
                if len(chunk.content) > 200:
                    preview += "..."

            # 显示结果
            self.console.print(f"\n{i}. [{score:.1%}] {doc_name}")
            if preview:
                self.console.print(f"   {preview}")

        self.console.print()

    async def _handle_refresh_command(self):
        """处理刷新命令"""
        if not self.monitor:
            self.console.print(Panel(
                "⚠️ 文件监控未启用或未配置目录。\n\n"
                "📝 启用文件监控:\n"
                "   1. 在 .env 文件中设置 MONITOR_DIRECTORIES\n"
                "   2. 示例: MONITOR_DIRECTORIES=/path/to/docs,/path/to/files\n"
                "   3. 重启会话\n\n"
                "💡 文件监控允许自动索引新文件和更新的文件。",
                title="[yellow]📁 文件监控不可用", border_style="yellow"
            ))
            return

        start_time = time.time()
        self.console.print("📁 正在刷新目录...")

        try:
            await self.monitor.refresh()
            refresh_time = time.time() - start_time

            self.console.print(Panel(
                f"✅ 目录刷新成功完成！\n"
                f"📄 所有新文件和更新的文件都已处理和索引。\n"
                f"⏱️ 刷新完成时间: {refresh_time:.2f} 秒",
                title="[green]🔄 刷新完成", border_style="green"
            ))

            logger.info(f"目录刷新在 {refresh_time:.2f}s 内完成")

        except Exception as e:
            logger.error(f"刷新失败: {e}")
            self.console.print(Panel(
                f"❌ 刷新过程中出错: {str(e)}\n\n"
                "🔧 故障排除步骤:\n"
                "   • 检查 MONITOR_DIRECTORIES 配置\n"
                "   • 验证目录权限\n"
                "   • 确保目录存在\n"
                "   • 检查磁盘空间可用性",
                title="[red]🔄 刷新失败", border_style="red"
            ))

    def _display_stats(self):
        """显示会话统计信息"""
        if not self.qa_agent:
            return

        session_duration = time.time() - self._session_start_time
        history_count = len(self.qa_agent.conversation_history.history)

        stats_text = f"""
📊 会话统计:
   • 会话时长: {session_duration:.1f} 秒
   • 开始时间: {datetime.fromtimestamp(self._session_start_time).strftime("%Y-%m-%d %H:%M:%S")}
   • 会话ID: {self.qa_agent.conversation_history.session_id}
   • 总问题数: {history_count}
   • 模型: {self.model or "默认"}
   • 数据库: {Path(self.db_path).name}
   • 监控状态: {"启用" if self.enable_monitoring else "禁用"}
"""
        self.console.print(Panel(stats_text, title="[bold bright_cyan]📈 性能仪表板", border_style="bright_cyan"))

    def _handle_save_command(self):
        """处理手动保存会话命令"""
        if not self.qa_agent or not self.qa_agent.conversation_history.history:
            self.console.print(Panel(
                "📝 还没有对话数据可保存。\n💡 开始提问以建立您的会话历史！",
                title="[yellow]💾 无内容可保存", border_style="yellow"
            ))
            return

        try:
            self.console.print("💾 正在保存会话...")
            session_file = self.qa_agent.conversation_history.save_to_file()

            self.console.print(Panel(
                f"✅ 会话保存成功！\n\n"
                f"📁 文件: {session_file}\n"
                f"📊 交换次数: {len(self.qa_agent.conversation_history.history)}\n"
                f"🆔 会话ID: {self.qa_agent.conversation_history.session_id}",
                title="[green]💾 保存完成", border_style="green"
            ))

        except Exception as e:
            logger.error(f"保存失败: {e}")
            self.console.print(Panel(
                f"❌ 保存会话失败: {str(e)}\n🔧 请检查文件权限和磁盘空间。",
                title="[red]💾 保存失败", border_style="red"
            ))

    def _display_history(self):
        """显示对话历史"""
        if not self.qa_agent.conversation_history.history:
            self.console.print(Panel(
                "📝 还没有对话历史。\n\n"
                "💡 开始提问以建立您的会话历史！\n"
                "🔄 您的对话上下文有助于我提供更好的答案。",
                title="[yellow]📜 对话历史", border_style="yellow"
            ))
            return

        self.console.print(f"\n📜 对话历史 ({len(self.qa_agent.conversation_history.history)} 次交换):")

        for i, exchange in enumerate(self.qa_agent.conversation_history.history, 1):
            timestamp = exchange.timestamp.strftime("%H:%M:%S")
            question_preview = exchange.question[:50] + "..." if len(exchange.question) > 50 else exchange.question
            answer_preview = exchange.answer[:50].replace('\n', ' ').strip()
            if len(exchange.answer) > 50:
                answer_preview += "..."

            source_count = len(exchange.search_results) if exchange.search_results else 0
            response_time = f"{exchange.response_time:.2f}s" if exchange.response_time > 0 else "-"

            self.console.print(f"\n{i}. [{timestamp}] {response_time}")
            self.console.print(f"   问: {question_preview}")
            self.console.print(f"   答: {answer_preview}")
            if source_count > 0:
                self.console.print(f"   来源: {source_count} 个文档")

        self.console.print()

    def _display_help(self):
        """显示帮助信息"""
        help_text = """
🎯 可用命令:
   • /help - 显示此帮助指南
   • /history - 显示对话历史
   • /clear - 清空对话历史
   • /search <查询> - 直接搜索文档
   • /refresh - 刷新监控目录
   • /stats - 显示会话统计
   • /save - 保存当前会话
   • /quit 或 /exit - 退出

💡 使用技巧:
   • 🧠 上下文感知: 我会记住我们的对话！可以自然地提出后续问题
   • 🎯 具体明确: 详细的问题会得到更好、更准确的答案
   • 🔍 先探索: 使用 /search 发现可用的文档和主题
   • 📚 来源透明: 我总是显示回答所依据的文档
   • 💾 保存会话: 使用 /save 保存重要的对话

⚙️ 技术规格:
   • 搜索算法: 混合 (向量 + 全文)
   • 上下文窗口: 最多 {self.config.max_context_length} 字符
   • 历史限制: 最近 {self.config.max_history} 次交换
   • 搜索结果: 前 {self.config.search_limit} 个文档

🚀 快速开始:
   1. 询问关于您文档的任何问题
   2. 使用 /search 探索可用内容
   3. 提出后续问题以获得更深入的见解
   4. 使用 /save 保存重要会话
"""
        self.console.print(Panel(help_text, title="[bold bright_yellow]📖 帮助", border_style="bright_yellow"))

    async def run(self):
        """运行交互式问答会话"""
        self._display_welcome()

        # 命令处理映射
        command_handlers = {
            "/help": self._display_help,
            "/history": self._display_history,
            "/clear": self._handle_clear_command,
            "/refresh": self._handle_refresh_command,
            "/stats": self._display_stats,
            "/save": self._handle_save_command,
        }

        while self._is_running:
            try:
                # 显示提示符
                question_count = len(self.qa_agent.conversation_history.history) if self.qa_agent else 0
                question = Prompt.ask(f"\n[bold bright_cyan]💭 问我任何问题 ({question_count} 个问题)[/bold bright_cyan]")

                if not question.strip():
                    continue

                # 处理退出命令
                if question.lower() in ["/quit", "/exit"]:
                    if self._handle_exit_command():
                        break
                    continue

                # 处理搜索命令
                if question.lower().startswith("/search "):
                    query = question[8:].strip()
                    await self._handle_search_command(query)
                    continue

                # 处理其他命令
                command = question.lower().strip()
                if command in command_handlers:
                    if asyncio.iscoroutinefunction(command_handlers[command]):
                        await command_handlers[command]()
                    else:
                        command_handlers[command]()
                    continue

                # 处理未知命令
                if question.startswith("/"):
                    self._handle_unknown_command(question)
                    continue

                # 处理常规问题
                await self._process_question(question)

            except KeyboardInterrupt:
                if self._handle_keyboard_interrupt():
                    break
                continue
            except Exception as e:
                self._handle_session_error(e)
                continue

    def _handle_clear_command(self):
        """处理清空命令"""
        if not self.qa_agent.conversation_history.history:
            self.console.print(Panel("📝 没有对话历史可清空。", title="[yellow]🧹 无内容可清空", border_style="yellow"))
            return

        # 请求确认
        confirm = Confirm.ask(f"[yellow]确定要清空 {len(self.qa_agent.conversation_history.history)} 次对话交换吗？[/yellow]")

        if confirm:
            self.qa_agent.conversation_history.clear()
            self.console.print(Panel(
                "🧹 对话历史已成功清空！\n✨ 重新开始，干净的状态。\n🧠 上下文记忆已重置。",
                title="[bold bright_green]✅ 历史已清空", border_style="bright_green"
            ))
            logger.info("用户清空了对话历史")
        else:
            self.console.print("[dim]清空操作已取消。[/dim]")

    def _handle_exit_command(self) -> bool:
        """处理退出命令"""
        # 显示会话摘要
        if self.qa_agent and self.qa_agent.conversation_history.history:
            session_duration = time.time() - self._session_start_time
            question_count = len(self.qa_agent.conversation_history.history)

            summary_text = f"""
📊 会话摘要:
   • 提问次数: {question_count}
   • 会话时长: {session_duration:.1f}s
   • 会话ID: {self.qa_agent.conversation_history.session_id}
"""
            self.console.print(Panel(summary_text, title="[bold bright_yellow]👋 会话完成！", border_style="bright_yellow"))

        # 最终告别消息
        farewell_text = """
🌟 感谢使用 HKEX ANNOUNCEMENT RAG！
💫 希望您找到了所需的答案。
🚀 随时回来获取更多见解！
"""
        self.console.print(Panel(farewell_text, title="[bold bright_yellow]🎉 再见！", border_style="bright_yellow"))

        return True

    def _handle_unknown_command(self, command: str):
        """处理未知命令"""
        available_commands = ["/help", "/history", "/clear", "/search", "/refresh", "/stats", "/save", "/quit", "/exit"]

        error_text = f"""
❌ 未知命令: {command}

💡 可用命令:
   • /help - 显示帮助
   • /history - 显示历史
   • /clear - 清空历史
   • /search <查询> - 搜索文档
   • /refresh - 刷新目录
   • /stats - 显示统计
   • /save - 保存会话
   • /quit 或 /exit - 退出

📖 输入 /help 查看完整命令参考
"""
        self.console.print(Panel(error_text, title="[red]⚠️ 命令未找到", border_style="red"))

    async def _process_question(self, question: str):
        """处理常规问题"""
        start_time = time.time()

        try:
            # 显示问题
            self._display_question(question)

            # 显示处理状态
            self.console.print("🚀 正在处理您的问题...")

            # 获取带上下文的答案
            answer, search_results = await self.qa_agent.answer_with_context(question)

            # 计算响应时间
            response_time = time.time() - start_time

            # 显示答案和响应时间
            self._display_answer(answer, search_results, response_time)

            logger.info(f"问题在 {response_time:.2f}s 内处理成功")

        except Exception as e:
            logger.error(f"处理问题时出错: {e}")

            error_text = f"""
❌ 处理您的问题时出错: {str(e)}

🔧 故障排除建议:
   • 尝试重新表述您的问题
   • 检查您的网络连接
   • 验证数据库是否可访问
   • 先尝试一个简单的问题

💡 输入 /help 获取指导或 /stats 查看系统状态
"""
            self.console.print(Panel(error_text, title="[red]⚠️ 处理错误", border_style="red"))

    def _handle_keyboard_interrupt(self) -> bool:
        """处理键盘中断"""
        self.console.print("\n⚠️ 会话被用户中断。")

        try:
            choice = Prompt.ask("[yellow]您想要做什么？ (继续/退出/保存)[/yellow]", default="继续")

            if choice.lower() in ['quit', 'exit', '退出', 'q']:
                return True
            elif choice.lower() in ['save', '保存', 's']:
                self._handle_save_command()
                return False
            else:
                self.console.print("[dim]继续会话...[/dim]")
                return False

        except KeyboardInterrupt:
            # 双重 Ctrl+C 意味着强制退出
            self.console.print("\n[red]检测到强制退出。正在退出...[/red]")
            return True

    def _handle_session_error(self, error: Exception):
        """处理一般会话错误"""
        logger.error(f"会话错误: {error}")

        error_text = f"""
❌ 会话错误: {str(error)}

🔧 故障排除:
   • 尝试重启会话
   • 检查系统资源
   • 验证配置设置

💡 会话将继续。输入 /help 获取帮助。
"""
        self.console.print(Panel(error_text, title="[red]⚠️ 会话错误", border_style="red"))


async def start_interactive_qa(db_path: str, model: str = "", enable_monitoring: bool = True,
        config: Optional[SessionConfig] = None) -> None:
    """
    启动交互式问答会话

    Args:
        db_path: 数据库文件路径
        model: 用于问答的模型名称
        enable_monitoring: 是否启用文件监控
        config: 可选的会话配置
    """
    console = Console()

    try:
        # 初始化配置
        session_config = config or SessionConfig()

        # 显示启动信息
        console.print(f"🚀 正在启动交互式问答会话...")
        console.print(f"📁 数据库: {Path(db_path).name}")
        console.print(f"🤖 模型: {model or '默认'}")
        console.print(f"📊 监控: {'启用' if enable_monitoring else '禁用'}")

        # 启动会话
        async with InteractiveQASession(db_path=db_path, model=model, enable_monitoring=enable_monitoring,
                config=session_config) as session:
            await session.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 会话被用户中断。再见！[/yellow]")
    except Exception as e:
        logger.error(f"启动交互式问答会话失败: {e}")
        console.print(f"[red]❌ 启动会话失败: {str(e)}[/red]")
        raise


async def interactive_qa_cli(db_path: str, model: str = "", enable_monitoring: bool = True,
        config_file: Optional[str] = None) -> None:
    """
    交互式问答的CLI包装器

    Args:
        db_path: 数据库文件路径
        model: 用于问答的模型名称
        enable_monitoring: 是否启用文件监控
        config_file: 可选的配置文件路径
    """
    console = Console()

    try:
        # 加载配置（如果提供）
        session_config = SessionConfig()
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                session_config = SessionConfig(**config_data)
                console.print(f"[green]✅ 从 {config_file} 加载配置[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ 加载配置文件失败: {e}。使用默认配置。[/yellow]")

        # 启动会话
        await start_interactive_qa(db_path=db_path, model=model, enable_monitoring=enable_monitoring,
            config=session_config)

    except Exception as e:
        logger.error(f"CLI错误: {e}")
        console.print(f"[red]❌ 启动交互式问答失败: {e}[/red]")


def create_session_config(**kwargs) -> SessionConfig:
    """创建带有自定义参数的会话配置"""
    return SessionConfig(**kwargs)


def load_session_from_file(session_file: Path) -> Optional[Dict[str, Any]]:
    """从文件加载保存的会话"""
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"从 {session_file} 加载会话失败: {e}")
        return None


if __name__ == "__main__":
    """简化的命令行界面"""
    import argparse

    parser = argparse.ArgumentParser(description="HKEX公告RAG交互式问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
示例:
  python interactive.py                          # 使用默认数据库
  python interactive.py mydata.db               # 使用指定数据库
  python interactive.py mydata.db gpt-4         # 使用指定模型
  python interactive.py --no-monitoring         # 禁用文件监控
  python interactive.py --config config.json    # 使用自定义配置
        """)

    parser.add_argument("db_path", nargs="?", default="haiku.rag.sqlite",
        help="数据库文件路径 (默认: haiku.rag.sqlite)")
    parser.add_argument("model", nargs="?", default="", help="用于问答的模型名称 (默认: 从配置)")
    parser.add_argument("--no-monitoring", action="store_true", help="禁用文件监控")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")

    args = parser.parse_args()

    # 配置日志
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("调试日志已启用")

    # 运行交互式会话
    try:
        asyncio.run(interactive_qa_cli(db_path=args.db_path, model=args.model,
            enable_monitoring=not args.no_monitoring, config_file=args.config))
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
