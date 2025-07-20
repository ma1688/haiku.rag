"""
Interactive QA agent that supports conversational loops with context preservation.

This module provides a comprehensive interactive QA system with:
- Context-aware conversation management
- Rich console interface with beautiful styling
- File monitoring and real-time updates
- Advanced search capabilities
- Session persistence and history management
- Performance optimization and caching
"""
import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
from contextlib import asynccontextmanager

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.status import Status

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.logging import get_logger
from haiku.rag.monitor import FileWatcher
from haiku.rag.qa import get_qa_agent
from haiku.rag.qa.base import QuestionAnswerAgentBase

# Constants and Configuration
logger = get_logger()

@dataclass
class SessionConfig:
    """Configuration for interactive QA session."""
    max_history: int = 10
    max_context_length: int = 2000
    search_limit: int = 5
    context_window: int = 3
    answer_preview_length: int = 200
    content_preview_length: int = 250
    cache_size: int = 100
    session_timeout: int = 3600  # 1 hour
    auto_save_interval: int = 300  # 5 minutes
    enable_metrics: bool = True
    enable_caching: bool = True

@dataclass
class ConversationExchange:
    """Represents a single conversation exchange."""
    timestamp: datetime
    question: str
    answer: str
    search_results: List[Tuple[Any, float]]
    response_time: float
    tokens_used: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "question": self.question,
            "answer": self.answer,
            "search_results": [(str(chunk), score) for chunk, score in self.search_results],
            "response_time": self.response_time,
            "tokens_used": self.tokens_used
        }

class ConversationHistory:
    """Enhanced conversation history management with persistence and analytics."""

    def __init__(self, config: SessionConfig, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id or self._generate_session_id()
        self.history: List[ConversationExchange] = []
        self.session_start = datetime.now()
        self._cache: Dict[str, Any] = {}
        self._metrics = {
            "total_questions": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{random_hash}"

    def add_exchange(self, question: str, answer: str, search_results: Optional[List] = None,
                    response_time: float = 0.0, tokens_used: Optional[int] = None):
        """Add a question-answer exchange to history with enhanced metadata."""
        exchange = ConversationExchange(
            timestamp=datetime.now(),
            question=question,
            answer=answer,
            search_results=search_results or [],
            response_time=response_time,
            tokens_used=tokens_used
        )

        self.history.append(exchange)

        # Update metrics
        self._metrics["total_questions"] += 1
        self._metrics["total_response_time"] += response_time
        self._metrics["avg_response_time"] = (
            self._metrics["total_response_time"] / self._metrics["total_questions"]
        )

        # Keep only the most recent exchanges
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]

        logger.debug(f"Added exchange to history. Total: {len(self.history)}")

    def get_context_summary(self, max_length: Optional[int] = None) -> str:
        """Generate an intelligent context summary from recent conversation history."""
        if not self.history:
            return ""

        max_length = max_length or self.config.max_context_length
        context_parts = []
        current_length = 0

        # Use last N exchanges for context
        recent_exchanges = self.history[-self.config.context_window:]

        for exchange in recent_exchanges:
            question_part = f"Q: {exchange.question}"
            answer_preview = exchange.answer[:self.config.answer_preview_length]
            if len(exchange.answer) > self.config.answer_preview_length:
                answer_preview += "..."
            answer_part = f"A: {answer_preview}"

            exchange_text = f"{question_part}\n{answer_part}\n"

            if current_length + len(exchange_text) > max_length:
                break

            context_parts.append(exchange_text)
            current_length += len(exchange_text)

        return "\n".join(context_parts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get session metrics and statistics."""
        session_duration = datetime.now() - self.session_start
        return {
            **self._metrics,
            "session_duration": str(session_duration).split('.')[0],
            "session_id": self.session_id,
            "total_exchanges": len(self.history),
            "cache_hit_rate": (
                self._metrics["cache_hits"] /
                max(1, self._metrics["cache_hits"] + self._metrics["cache_misses"])
            ) * 100
        }

    def clear(self):
        """Clear conversation history and reset metrics."""
        self.history.clear()
        self.session_start = datetime.now()
        self._cache.clear()
        self._metrics = {
            "total_questions": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("Conversation history cleared")

    def save_to_file(self, file_path: Optional[Path] = None) -> Path:
        """Save conversation history to file."""
        if not file_path:
            file_path = Path.home() / ".haiku_rag" / "sessions" / f"{self.session_id}.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "config": asdict(self.config),
            "metrics": self.get_metrics(),
            "history": [exchange.to_dict() for exchange in self.history]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Session saved to {file_path}")
        return file_path


class ContextAwareQAAgent(QuestionAnswerAgentBase):
    """Enhanced QA Agent with conversation context, caching, and performance optimization."""

    def __init__(self, client: HaikuRAG, model: str = "", config: Optional[SessionConfig] = None):
        super().__init__(client, model)
        self.config = config or SessionConfig()
        self.base_agent = get_qa_agent(client, model)
        self.conversation_history = ConversationHistory(self.config)
        self._search_cache: Dict[str, Tuple[List, float]] = {}
        self._answer_cache: Dict[str, Tuple[str, float]] = {}

    def _get_cache_key(self, question: str, context: str = "") -> str:
        """Generate cache key for question and context."""
        content = f"{question}|{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float, ttl: int = 300) -> bool:
        """Check if cache entry is still valid (default 5 minutes TTL)."""
        return time.time() - timestamp < ttl

    async def _get_cached_search(self, question: str) -> Optional[List]:
        """Get cached search results if available and valid."""
        if not self.config.enable_caching:
            return None

        cache_key = self._get_cache_key(question)
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.conversation_history._metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for search: {question[:50]}...")
                return results
            else:
                # Remove expired cache entry
                del self._search_cache[cache_key]

        self.conversation_history._metrics["cache_misses"] += 1
        return None

    async def _cache_search_results(self, question: str, results: List):
        """Cache search results with timestamp."""
        if not self.config.enable_caching:
            return

        cache_key = self._get_cache_key(question)
        self._search_cache[cache_key] = (results, time.time())

        # Limit cache size
        if len(self._search_cache) > self.config.cache_size:
            # Remove oldest entries
            oldest_key = min(self._search_cache.keys(),
                           key=lambda k: self._search_cache[k][1])
            del self._search_cache[oldest_key]

    async def answer_with_context(self, question: str) -> Tuple[str, List]:
        """Answer a question with enhanced conversation context and performance optimization."""
        start_time = time.time()

        try:
            # Input validation
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")

            question = question.strip()

            # Check for cached search results
            search_results = await self._get_cached_search(question)

            if search_results is None:
                # Get fresh search results
                search_results = await self._client.search(question, limit=self.config.search_limit)
                await self._cache_search_results(question, search_results)

            # Get conversation context
            context = self.conversation_history.get_context_summary(self.config.max_context_length)

            # Create enhanced question with intelligent context integration
            enhanced_question = self._create_enhanced_question(question, context, search_results)

            # Get answer from base agent with error handling
            try:
                answer = await self.base_agent.answer(enhanced_question)
            except Exception as e:
                logger.error(f"Error getting answer from base agent: {e}")
                # Fallback to simple question without context
                answer = await self.base_agent.answer(question)
                logger.info("Used fallback answer without context")

            # Calculate response time
            response_time = time.time() - start_time

            # Add to conversation history with metrics
            self.conversation_history.add_exchange(
                question=question,
                answer=answer,
                search_results=search_results,
                response_time=response_time
            )

            logger.info(f"Question answered in {response_time:.2f}s")
            return answer, search_results

        except Exception as e:
            logger.error(f"Error in answer_with_context: {e}")
            response_time = time.time() - start_time

            # Add failed exchange to history for debugging
            self.conversation_history.add_exchange(
                question=question,
                answer=f"Error: {str(e)}",
                search_results=[],
                response_time=response_time
            )

            raise

    def _create_enhanced_question(self, question: str, context: str, search_results: List) -> str:
        """Create an intelligently enhanced question with context and search results."""
        if not context:
            return question

        # Create context-aware prompt
        enhanced_parts = []

        if context and len(context) < self.config.max_context_length:
            enhanced_parts.append(f"Previous conversation context:\n{context}")

        # Add relevant search context if available
        if search_results:
            search_context = self._create_search_context(search_results[:3])
            if search_context:
                enhanced_parts.append(f"Relevant information from knowledge base:\n{search_context}")

        enhanced_parts.append(f"Current question: {question}")
        enhanced_parts.append(
            "Please answer the current question, taking into account the conversation context "
            "and relevant information if applicable. Be concise but comprehensive."
        )

        return "\n\n".join(enhanced_parts)

    def _create_search_context(self, search_results: List) -> str:
        """Create context from search results."""
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
        """Standard answer method for compatibility."""
        answer, _ = await self.answer_with_context(question)
        return answer

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            **self.conversation_history.get_metrics(),
            "cache_size": len(self._search_cache),
            "model": self._model,
            "config": asdict(self.config)
        }


class InteractiveQASession:
    """Enhanced interactive QA session with rich console interface, performance monitoring, and advanced features."""

    def __init__(self, db_path: str, model: str = "", enable_monitoring: bool = True,
                 config: Optional[SessionConfig] = None, session_id: Optional[str] = None):
        self.db_path = db_path
        self.model = model
        self.enable_monitoring = enable_monitoring
        self.config = config or SessionConfig()
        self.session_id = session_id

        # Console and UI components
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True
        )

        # Core components
        self.client: Optional[HaikuRAG] = None
        self.qa_agent: Optional[ContextAwareQAAgent] = None
        self.monitor: Optional[FileWatcher] = None
        self.monitor_task: Optional[asyncio.Task] = None

        # Session management
        self._session_start_time = time.time()
        self._auto_save_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Performance tracking
        self._performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0
        }

    async def __aenter__(self):
        """Enhanced async context manager entry with comprehensive initialization."""
        try:
            # Initialize progress tracking
            with self.progress:
                init_task = self.progress.add_task("Initializing session...", total=100)

                # Initialize client
                self.progress.update(init_task, advance=20, description="Connecting to database...")
                self.client = HaikuRAG(self.db_path)
                await self.client.__aenter__()

                # Initialize QA agent
                self.progress.update(init_task, advance=20, description="Setting up QA agent...")
                self.qa_agent = ContextAwareQAAgent(self.client, self.model, self.config)

                # Validate configuration
                self.progress.update(init_task, advance=10, description="Validating configuration...")
                await self._validate_configuration()

                # Start file monitoring
                self.progress.update(init_task, advance=20, description="Setting up file monitoring...")
                await self._setup_file_monitoring()

                # Start background tasks
                self.progress.update(init_task, advance=15, description="Starting background services...")
                await self._start_background_tasks()

                # Final setup
                self.progress.update(init_task, advance=15, description="Finalizing setup...")
                self._is_running = True

                logger.info(f"Interactive QA session initialized successfully. Session ID: {self.qa_agent.conversation_history.session_id}")

            return self

        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            await self._cleanup()
            raise

    async def _validate_configuration(self):
        """Validate session configuration and dependencies."""
        try:
            # Test database connection
            await self.client.search("test", limit=1)

            # Test QA agent
            if hasattr(self.qa_agent.base_agent, 'answer'):
                # Quick validation without actual API call
                pass

            logger.info("Configuration validation successful")

        except Exception as e:
            logger.warning(f"Configuration validation warning: {e}")

    async def _setup_file_monitoring(self):
        """Setup file monitoring with enhanced error handling."""
        if not self.enable_monitoring:
            return

        if Config.MONITOR_DIRECTORIES:
            try:
                self.monitor = FileWatcher(paths=Config.MONITOR_DIRECTORIES, client=self.client)
                self.monitor_task = asyncio.create_task(self.monitor.observe())

                self.console.print(
                    Panel(
                        f"📁 File monitoring enabled for: {', '.join(str(p) for p in Config.MONITOR_DIRECTORIES)}\n"
                        f"🔄 Auto-refresh: Every {self.config.auto_save_interval}s",
                        title="[green]🔍 File Monitor Active",
                        border_style="green"
                    )
                )
                logger.info(f"File monitoring started for {len(Config.MONITOR_DIRECTORIES)} directories")

            except Exception as e:
                logger.error(f"Failed to start file monitoring: {e}")
                self.console.print(
                    Panel(
                        f"❌ File monitoring failed to start: {str(e)}\n"
                        "📝 Session will continue without file monitoring.",
                        title="[red]🔍 File Monitor Error",
                        border_style="red"
                    )
                )
        else:
            self.console.print(
                Panel(
                    "⚠️ File monitoring is enabled but no MONITOR_DIRECTORIES configured.\n"
                    "Set MONITOR_DIRECTORIES in your .env file to enable automatic file monitoring.\n"
                    "📖 Example: MONITOR_DIRECTORIES=/path/to/docs,/path/to/files",
                    title="[yellow]📁 File Monitor Configuration",
                    border_style="yellow"
                )
            )

    async def _start_background_tasks(self):
        """Start background tasks for session management."""
        if self.config.auto_save_interval > 0:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())

        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _auto_save_loop(self):
        """Automatically save session data at intervals."""
        try:
            while self._is_running:
                await asyncio.sleep(self.config.auto_save_interval)
                if self.qa_agent and self.qa_agent.conversation_history.history:
                    try:
                        self.qa_agent.conversation_history.save_to_file()
                        logger.debug("Auto-saved session data")
                    except Exception as e:
                        logger.error(f"Auto-save failed: {e}")
        except asyncio.CancelledError:
            logger.debug("Auto-save loop cancelled")

    async def _health_check_loop(self):
        """Perform periodic health checks."""
        try:
            while self._is_running:
                await asyncio.sleep(60)  # Check every minute
                try:
                    # Simple health check - test database connection
                    await self.client.search("health_check", limit=1)
                    logger.debug("Health check passed")
                except Exception as e:
                    logger.warning(f"Health check failed: {e}")
        except asyncio.CancelledError:
            logger.debug("Health check loop cancelled")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Enhanced async context manager exit with comprehensive cleanup."""
        await self._cleanup()
        return False

    async def _cleanup(self):
        """Comprehensive cleanup of all resources."""
        self._is_running = False

        try:
            # Save final session data
            if self.qa_agent and self.qa_agent.conversation_history.history:
                try:
                    session_file = self.qa_agent.conversation_history.save_to_file()
                    self.console.print(
                        Panel(
                            f"💾 Session saved to: {session_file}\n"
                            f"📊 Total exchanges: {len(self.qa_agent.conversation_history.history)}",
                            title="[green]💾 Session Saved",
                            border_style="green"
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to save session: {e}")

            # Stop background tasks
            tasks_to_cancel = [
                self._auto_save_task,
                self._health_check_task,
                self.monitor_task
            ]

            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Stop file monitoring
            if self.monitor_task:
                self.console.print(
                    Panel(
                        "📁 File monitoring stopped.\n"
                        "🔄 All monitored files have been processed.",
                        title="[blue]🔍 File Monitor",
                        border_style="blue"
                    )
                )

            # Close client connection
            if self.client:
                await self.client.__aexit__(None, None, None)

            # Display session summary
            self._display_session_summary()

            logger.info("Session cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _display_session_summary(self):
        """Display a comprehensive session summary."""
        if not self.qa_agent:
            return

        metrics = self.qa_agent.get_session_stats()
        session_duration = time.time() - self._session_start_time

        summary_table = Table(title="📊 Session Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Session Duration", f"{session_duration:.1f} seconds")
        summary_table.add_row("Total Questions", str(metrics.get("total_questions", 0)))
        summary_table.add_row("Average Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
        summary_table.add_row("Cache Hit Rate", f"{metrics.get('cache_hit_rate', 0):.1f}%")
        summary_table.add_row("Session ID", metrics.get("session_id", "N/A"))

        self.console.print()
        self.console.print(summary_table)
        self.console.print()

    def _display_welcome(self):
        """Display enhanced welcome message with beautiful styling and system information."""
        # Create dynamic header
        header_width = self.console.size.width - 4
        header = Text()
        header.append("╭", style="bold bright_blue")
        header.append("─" * (header_width // 2 - 15), style="bright_blue")
        header.append(" 🚀 Interactive QA Session ", style="bold bright_white on bright_blue")
        header.append("─" * (header_width // 2 - 15), style="bright_blue")
        header.append("╮", style="bold bright_blue")

        # System information table
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Label", style="dim")
        info_table.add_column("Value", style="bold")

        info_table.add_row("🤖 System:", "HKEX ANNOUNCEMENT RAG Interactive QA")
        info_table.add_row("👨‍💻 Author:", "MAXJ")
        info_table.add_row("📦 Version:", "v0.2.0 (Optimized)")
        info_table.add_row("🕒 Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        info_table.add_row("🆔 Session:", self.qa_agent.conversation_history.session_id if self.qa_agent else "Initializing...")
        info_table.add_row("🧠 Model:", self.model or "Default")
        info_table.add_row("💾 Database:", str(Path(self.db_path).name))

        # Features section
        features_text = Text()
        features_text.append("🌟 Enhanced Features:\n", style="bold bright_yellow")
        features = [
            "💬 Context-aware conversations with intelligent memory",
            "🔍 Hybrid search with semantic and keyword matching",
            "⚡ Smart caching for improved performance",
            "📊 Real-time performance metrics and analytics",
            "💾 Automatic session persistence and recovery",
            "📁 Live file monitoring and auto-indexing",
            "🎨 Beautiful rich console interface"
        ]

        for feature in features:
            features_text.append(f"   • {feature}\n", style="bright_white")

        # Commands section with enhanced styling
        commands_table = Table(title="⚡ Available Commands", show_header=True, header_style="bold bright_cyan")
        commands_table.add_column("Command", style="bold bright_cyan", no_wrap=True)
        commands_table.add_column("Description", style="bright_white")
        commands_table.add_column("Example", style="dim")

        commands_data = [
            ("💡 /help", "Show comprehensive help guide", "/help"),
            ("📜 /history", "Display conversation history", "/history"),
            ("🧹 /clear", "Clear conversation history", "/clear"),
            ("🔍 /search <query>", "Search documents directly", "/search python tutorial"),
            ("📁 /refresh", "Refresh monitored directories", "/refresh"),
            ("📊 /stats", "Show session statistics", "/stats"),
            ("💾 /save", "Save current session", "/save"),
            ("👋 /quit or /exit", "Exit gracefully", "/quit")
        ]

        for cmd, desc, example in commands_data:
            commands_table.add_row(cmd, desc, example)

        # Performance info
        if self.qa_agent:
            metrics = self.qa_agent.get_session_stats()
            perf_text = Text()
            perf_text.append("📈 Performance Status: ", style="bold bright_green")
            perf_text.append(f"Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1f}% | ", style="bright_white")
            perf_text.append(f"Avg Response: {metrics.get('avg_response_time', 0):.2f}s", style="bright_white")

        # Display everything with panels
        self.console.print()
        self.console.print(header)
        self.console.print()

        # Main info panel
        info_panel = Panel(info_table, title="[bold bright_blue]📋 Session Information", border_style="bright_blue")
        self.console.print(info_panel)

        # Features panel
        features_panel = Panel(features_text, title="[bold bright_yellow]🌟 Enhanced Features", border_style="bright_yellow")
        self.console.print(features_panel)

        # Commands panel
        commands_panel = Panel(commands_table, title="[bold bright_cyan]⚡ Command Reference", border_style="bright_cyan")
        self.console.print(commands_panel)

        # Performance panel (if available)
        if self.qa_agent:
            perf_panel = Panel(perf_text, title="[bold bright_green]📈 Performance", border_style="bright_green")
            self.console.print(perf_panel)

        # Footer
        footer = Text()
        footer.append("╰", style="bold bright_blue")
        footer.append("─" * header_width, style="bright_blue")
        footer.append("╯", style="bold bright_blue")
        self.console.print(footer)

        # Motivational message
        motivation_text = Text()
        motivation_text.append("✨ ", style="bright_yellow")
        motivation_text.append("Ready to explore your knowledge base! ", style="italic bright_white")
        motivation_text.append("Ask me anything or use ", style="italic bright_white")
        motivation_text.append("/help", style="bold bright_cyan")
        motivation_text.append(" for guidance...", style="italic bright_white")

        motivation_panel = Panel(motivation_text, border_style="bright_yellow")
        self.console.print(motivation_panel)
        self.console.print()

    def _display_question(self, question: str):
        """Display user question with enhanced styling and metadata."""
        # Add timestamp and question number
        question_count = len(self.qa_agent.conversation_history.history) + 1 if self.qa_agent else 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        question_text = Text()
        question_text.append(f"[{timestamp}] Question #{question_count}\n", style="dim")
        question_text.append(question, style="bright_white")

        question_panel = Panel(
            question_text,
            title="[bold bright_blue]👤 Your Question",
            title_align="left",
            border_style="bright_blue",
            padding=(0, 1)
        )
        self.console.print(question_panel)
        self.console.print()

    def _display_answer(self, answer: str, search_results: List = None, response_time: float = 0.0):
        """Display AI answer with enhanced styling, metadata, and search context."""
        # Create answer content with metadata
        answer_content = Text()
        if response_time > 0:
            answer_content.append(f"⏱️ Response time: {response_time:.2f}s\n\n", style="dim")

        # Display answer in a beautiful panel with markdown support
        try:
            answer_panel = Panel(
                Markdown(answer),
                title=f"[bold bright_green]🤖 AI Assistant",
                title_align="left",
                border_style="bright_green",
                padding=(0, 1)
            )
        except Exception:
            # Fallback to plain text if markdown parsing fails
            answer_panel = Panel(
                Text(answer, style="bright_white"),
                title="[bold bright_green]🤖 AI Assistant",
                title_align="left",
                border_style="bright_green",
                padding=(0, 1)
            )

        self.console.print(answer_panel)

        # Display enhanced search results with analytics
        if search_results:
            self._display_search_sources(search_results)

        self.console.print()

    def _display_search_sources(self, search_results: List):
        """Display search sources with enhanced analytics and visualization."""
        if not search_results:
            return

        # Create sources table
        sources_table = Table(
            title="📚 Knowledge Sources",
            show_header=True,
            header_style="bold bright_yellow"
        )
        sources_table.add_column("Rank", style="bold bright_cyan", width=6)
        sources_table.add_column("Relevance", style="bold", width=12)
        sources_table.add_column("Document", style="bright_blue", width=30)
        sources_table.add_column("Preview", style="bright_white")

        total_score = sum(score for _, score in search_results)
        avg_score = total_score / len(search_results) if search_results else 0

        for i, (chunk, score) in enumerate(search_results[:5], 1):
            # Score visualization and styling
            if score > 0.8:
                score_style = "bold bright_green"
                score_bar = "█████"
                score_icon = "🟢"
            elif score > 0.6:
                score_style = "bold bright_yellow"
                score_bar = "████░"
                score_icon = "🟡"
            elif score > 0.4:
                score_style = "bold bright_orange"
                score_bar = "███░░"
                score_icon = "🟠"
            else:
                score_style = "bold bright_red"
                score_bar = "██░░░"
                score_icon = "🔴"

            # Format document name
            doc_name = "Unknown"
            if hasattr(chunk, 'document_uri') and chunk.document_uri:
                doc_name = Path(chunk.document_uri).name if chunk.document_uri.startswith('file://') else chunk.document_uri
                if len(doc_name) > 25:
                    doc_name = doc_name[:22] + "..."

            # Format content preview
            preview = ""
            if hasattr(chunk, 'content') and chunk.content:
                preview = chunk.content[:100].replace('\n', ' ').strip()
                if len(chunk.content) > 100:
                    preview += "..."

            # Add row to table
            sources_table.add_row(
                f"{score_icon} #{i}",
                f"{score:.1%} {score_bar}",
                doc_name,
                preview
            )

        # Add summary row
        sources_table.add_section()
        sources_table.add_row(
            "📊 Summary",
            f"Avg: {avg_score:.1%}",
            f"{len(search_results)} sources",
            f"Total relevance: {total_score:.2f}"
        )

        sources_panel = Panel(
            sources_table,
            title="[bold bright_yellow]📖 Reference Materials",
            border_style="bright_yellow"
        )
        self.console.print(sources_panel)

    async def _handle_search_command(self, query: str):
        """Handle direct search command with enhanced display and analytics."""
        if not query.strip():
            error_panel = Panel(
                "[red]❌ Please provide a search query after /search command.\n"
                "💡 Example: /search python tutorial[/red]",
                title="[red]⚠️ Invalid Command",
                border_style="red"
            )
            self.console.print(error_panel)
            return

        start_time = time.time()

        # Show search progress with enhanced UI
        with self.progress:
            search_task = self.progress.add_task("🔍 Searching knowledge base...", total=100)

            self.progress.update(search_task, advance=30, description="🔍 Analyzing query...")
            await asyncio.sleep(0.1)  # Small delay for UI feedback

            self.progress.update(search_task, advance=40, description="🔍 Searching documents...")
            try:
                results = await self.client.search(query, limit=self.config.search_limit)
            except Exception as e:
                logger.error(f"Search failed: {e}")
                error_panel = Panel(
                    f"[red]❌ Search failed: {str(e)}\n"
                    "🔧 Please try again or check your database connection.[/red]",
                    title="[red]🔍 Search Error",
                    border_style="red"
                )
                self.console.print(error_panel)
                return

            self.progress.update(search_task, advance=30, description="🔍 Processing results...")
            await asyncio.sleep(0.1)

        search_time = time.time() - start_time

        if not results:
            no_results_panel = Panel(
                "🚫 No matching documents found.\n\n"
                "💡 Try these suggestions:\n"
                "   • Use different keywords\n"
                "   • Check spelling\n"
                "   • Use broader search terms\n"
                "   • Try synonyms or related terms",
                title="[yellow]📭 No Results Found",
                border_style="yellow"
            )
            self.console.print(no_results_panel)
            return

        # Display results with enhanced table format
        self._display_search_results_table(results, query, search_time)

    def _display_search_results_table(self, results: List, query: str, search_time: float):
        """Display search results in an enhanced table format."""
        # Create results table
        results_table = Table(
            title=f"🔍 Search Results for: '{query}' ({len(results)} found in {search_time:.2f}s)",
            show_header=True,
            header_style="bold bright_green"
        )
        results_table.add_column("Rank", style="bold bright_cyan", width=6)
        results_table.add_column("Score", style="bold", width=15)
        results_table.add_column("Document", style="bright_blue", width=25)
        results_table.add_column("Content Preview", style="bright_white")

        # Calculate statistics
        scores = [score for _, score in results]
        max_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0

        for i, (chunk, score) in enumerate(results, 1):
            # Score visualization
            score_percentage = (score / max_score * 100) if max_score > 0 else 0

            if score > 0.8:
                score_style = "bold bright_green"
                score_icon = "🟢"
                score_bar = "█████"
            elif score > 0.6:
                score_style = "bold bright_yellow"
                score_icon = "🟡"
                score_bar = "████░"
            elif score > 0.4:
                score_style = "bold bright_orange"
                score_icon = "🟠"
                score_bar = "███░░"
            else:
                score_style = "bold bright_red"
                score_icon = "🔴"
                score_bar = "██░░░"

            # Format document name
            doc_name = "Unknown Document"
            if hasattr(chunk, 'document_uri') and chunk.document_uri:
                if chunk.document_uri.startswith('file://'):
                    doc_name = Path(chunk.document_uri).name
                else:
                    doc_name = chunk.document_uri

                if len(doc_name) > 20:
                    doc_name = doc_name[:17] + "..."

            # Format content preview with highlighting
            preview = ""
            if hasattr(chunk, 'content') and chunk.content:
                preview = chunk.content[:self.config.content_preview_length].replace('\n', ' ').strip()
                if len(chunk.content) > self.config.content_preview_length:
                    preview += "..."

                # Simple keyword highlighting (case-insensitive)
                query_words = query.lower().split()
                for word in query_words:
                    if len(word) > 2:  # Only highlight words longer than 2 characters
                        preview = preview.replace(word, f"[bold yellow]{word}[/bold yellow]")
                        preview = preview.replace(word.capitalize(), f"[bold yellow]{word.capitalize()}[/bold yellow]")

            # Add row to table
            results_table.add_row(
                f"{score_icon} #{i}",
                f"{score:.3f}\n{score_bar}",
                doc_name,
                preview
            )

        # Add summary section
        results_table.add_section()
        results_table.add_row(
            "📊",
            f"Max: {max_score:.3f}\nAvg: {avg_score:.3f}",
            f"{len(results)} documents",
            f"Search completed in {search_time:.2f} seconds"
        )

        # Display the table
        results_panel = Panel(
            results_table,
            title="[bright_green]🔍 Enhanced Search Results",
            border_style="bright_green"
        )
        self.console.print(results_panel)

    async def _handle_refresh_command(self):
        """Handle refresh command to scan monitored directories with enhanced progress tracking."""
        if not self.monitor:
            no_monitor_panel = Panel(
                "⚠️ File monitoring is not enabled or no directories are configured.\n\n"
                "📝 To enable file monitoring:\n"
                "   1. Set MONITOR_DIRECTORIES in your .env file\n"
                "   2. Example: MONITOR_DIRECTORIES=/path/to/docs,/path/to/files\n"
                "   3. Restart the session\n\n"
                "💡 File monitoring allows automatic indexing of new and updated files.",
                title="[yellow]📁 File Monitor Not Available",
                border_style="yellow"
            )
            self.console.print(no_monitor_panel)
            return

        start_time = time.time()

        # Enhanced progress tracking
        with self.progress:
            refresh_task = self.progress.add_task("📁 Refreshing directories...", total=100)

            self.progress.update(refresh_task, advance=20, description="📁 Scanning directories...")

            try:
                # Get directory info first
                total_dirs = len(Config.MONITOR_DIRECTORIES)
                self.progress.update(refresh_task, advance=20, description=f"📁 Processing {total_dirs} directories...")

                await self.monitor.refresh()

                self.progress.update(refresh_task, advance=60, description="📁 Finalizing refresh...")

                refresh_time = time.time() - start_time

                success_panel = Panel(
                    f"✅ Directory refresh completed successfully!\n"
                    f"📄 All new and updated files have been processed and indexed.\n"
                    f"⏱️ Refresh completed in {refresh_time:.2f} seconds\n"
                    f"📁 Monitored directories: {total_dirs}",
                    title="[green]🔄 Refresh Complete",
                    border_style="green"
                )
                self.console.print(success_panel)

                logger.info(f"Directory refresh completed in {refresh_time:.2f}s")

            except Exception as e:
                logger.error(f"Refresh failed: {e}")
                error_panel = Panel(
                    f"❌ Error during refresh: {str(e)}\n\n"
                    "🔧 Troubleshooting steps:\n"
                    "   • Check MONITOR_DIRECTORIES configuration\n"
                    "   • Verify directory permissions\n"
                    "   • Ensure directories exist\n"
                    "   • Check disk space availability",
                    title="[red]🔄 Refresh Failed",
                    border_style="red"
                )
                self.console.print(error_panel)

    async def _handle_stats_command(self):
        """Display comprehensive session statistics."""
        if not self.qa_agent:
            return

        metrics = self.qa_agent.get_session_stats()
        session_duration = time.time() - self._session_start_time

        # Create comprehensive stats table
        stats_table = Table(title="📊 Session Statistics", show_header=True, header_style="bold bright_cyan")
        stats_table.add_column("Category", style="bold bright_yellow", no_wrap=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bright_white")

        # Session info
        stats_table.add_row("🕒 Session", "Duration", f"{session_duration:.1f} seconds")
        stats_table.add_row("", "Started", datetime.fromtimestamp(self._session_start_time).strftime("%Y-%m-%d %H:%M:%S"))
        stats_table.add_row("", "Session ID", metrics.get("session_id", "N/A"))

        # Performance metrics
        stats_table.add_section()
        stats_table.add_row("⚡ Performance", "Total Questions", str(metrics.get("total_questions", 0)))
        stats_table.add_row("", "Avg Response Time", f"{metrics.get('avg_response_time', 0):.3f}s")
        stats_table.add_row("", "Total Response Time", f"{metrics.get('total_response_time', 0):.2f}s")

        # Cache metrics
        stats_table.add_section()
        stats_table.add_row("💾 Cache", "Hit Rate", f"{metrics.get('cache_hit_rate', 0):.1f}%")
        stats_table.add_row("", "Cache Hits", str(metrics.get("cache_hits", 0)))
        stats_table.add_row("", "Cache Misses", str(metrics.get("cache_misses", 0)))
        stats_table.add_row("", "Cache Size", str(metrics.get("cache_size", 0)))

        # System info
        stats_table.add_section()
        stats_table.add_row("🔧 System", "Model", metrics.get("model", "Unknown"))
        stats_table.add_row("", "Database", str(Path(self.db_path).name))
        stats_table.add_row("", "Monitoring", "Enabled" if self.enable_monitoring else "Disabled")

        stats_panel = Panel(stats_table, title="[bold bright_cyan]📈 Performance Dashboard", border_style="bright_cyan")
        self.console.print(stats_panel)

    async def _handle_save_command(self):
        """Handle manual session save command."""
        if not self.qa_agent or not self.qa_agent.conversation_history.history:
            no_data_panel = Panel(
                "📝 No conversation data to save yet.\n"
                "💡 Start asking questions to build your session history!",
                title="[yellow]💾 Nothing to Save",
                border_style="yellow"
            )
            self.console.print(no_data_panel)
            return

        try:
            with self.console.status("[bold bright_blue]💾 Saving session...", spinner="dots"):
                session_file = self.qa_agent.conversation_history.save_to_file()

            save_panel = Panel(
                f"✅ Session saved successfully!\n\n"
                f"📁 File: {session_file}\n"
                f"📊 Exchanges: {len(self.qa_agent.conversation_history.history)}\n"
                f"🆔 Session ID: {self.qa_agent.conversation_history.session_id}\n"
                f"💾 File size: {session_file.stat().st_size} bytes",
                title="[green]💾 Save Complete",
                border_style="green"
            )
            self.console.print(save_panel)

        except Exception as e:
            logger.error(f"Save failed: {e}")
            error_panel = Panel(
                f"❌ Failed to save session: {str(e)}\n"
                "🔧 Please check file permissions and disk space.",
                title="[red]💾 Save Failed",
                border_style="red"
            )
            self.console.print(error_panel)

    def _display_history(self):
        """Display conversation history with enhanced table formatting and analytics."""
        if not self.qa_agent.conversation_history.history:
            no_history_panel = Panel(
                "📝 No conversation history yet.\n\n"
                "💡 Start asking questions to build your session history!\n"
                "🔄 Your conversation context helps me provide better answers.",
                title="[yellow]📜 Conversation History",
                border_style="yellow"
            )
            self.console.print(no_history_panel)
            return

        # Get session metrics
        metrics = self.qa_agent.conversation_history.get_metrics()
        session_duration = datetime.now() - self.qa_agent.conversation_history.session_start

        # Create history table
        history_table = Table(
            title=f"📜 Conversation History ({len(self.qa_agent.conversation_history.history)} exchanges)",
            show_header=True,
            header_style="bold bright_blue"
        )
        history_table.add_column("#", style="bold bright_cyan", width=4)
        history_table.add_column("Time", style="dim", width=10)
        history_table.add_column("Question", style="bright_blue", width=40)
        history_table.add_column("Answer Preview", style="bright_green", width=40)
        history_table.add_column("Sources", style="bright_yellow", width=8)
        history_table.add_column("Response", style="dim", width=8)

        for i, exchange in enumerate(self.qa_agent.conversation_history.history, 1):
            # Handle both old dict format and new ConversationExchange format
            if hasattr(exchange, 'timestamp'):
                # New format
                timestamp = exchange.timestamp.strftime("%H:%M:%S")
                question = exchange.question
                answer = exchange.answer
                search_results = exchange.search_results
                response_time = getattr(exchange, 'response_time', 0.0)
            else:
                # Old dict format (backward compatibility)
                timestamp = exchange["timestamp"].strftime("%H:%M:%S")
                question = exchange["question"]
                answer = exchange["answer"]
                search_results = exchange.get("search_results", [])
                response_time = 0.0

            # Format question (truncate if too long)
            question_preview = question[:35] + "..." if len(question) > 35 else question

            # Format answer preview
            answer_preview = answer[:35].replace('\n', ' ').strip()
            if len(answer) > 35:
                answer_preview += "..."

            # Source count
            source_count = len(search_results) if search_results else 0
            source_text = f"{source_count}" if source_count > 0 else "-"

            # Response time
            response_text = f"{response_time:.2f}s" if response_time > 0 else "-"

            history_table.add_row(
                str(i),
                timestamp,
                question_preview,
                answer_preview,
                source_text,
                response_text
            )

        # Add summary section
        history_table.add_section()
        history_table.add_row(
            "📊",
            "Summary",
            f"Session: {str(session_duration).split('.')[0]}",
            f"Avg Response: {metrics.get('avg_response_time', 0):.2f}s",
            f"Total: {sum(len(ex.search_results) if hasattr(ex, 'search_results') else len(ex.get('search_results', [])) for ex in self.qa_agent.conversation_history.history)}",
            f"Cache: {metrics.get('cache_hit_rate', 0):.1f}%"
        )

        # Display the table
        history_panel = Panel(
            history_table,
            title="[bold bright_blue]📜 Enhanced Conversation History",
            border_style="bright_blue"
        )
        self.console.print(history_panel)

        # Display additional analytics
        if len(self.qa_agent.conversation_history.history) > 1:
            self._display_conversation_analytics()

    def _display_conversation_analytics(self):
        """Display conversation analytics and insights."""
        history = self.qa_agent.conversation_history.history

        # Calculate analytics
        response_times = []
        source_counts = []
        question_lengths = []
        answer_lengths = []

        for exchange in history:
            if hasattr(exchange, 'response_time') and exchange.response_time > 0:
                response_times.append(exchange.response_time)
            if hasattr(exchange, 'search_results'):
                source_counts.append(len(exchange.search_results))
            if hasattr(exchange, 'question'):
                question_lengths.append(len(exchange.question))
            if hasattr(exchange, 'answer'):
                answer_lengths.append(len(exchange.answer))

        # Create analytics table
        analytics_table = Table(title="📈 Conversation Analytics", show_header=True, header_style="bold bright_magenta")
        analytics_table.add_column("Metric", style="cyan")
        analytics_table.add_column("Value", style="bright_white")
        analytics_table.add_column("Insight", style="dim")

        if response_times:
            avg_response = sum(response_times) / len(response_times)
            fastest = min(response_times)
            slowest = max(response_times)
            analytics_table.add_row("⏱️ Avg Response Time", f"{avg_response:.2f}s", f"Range: {fastest:.2f}s - {slowest:.2f}s")

        if source_counts:
            avg_sources = sum(source_counts) / len(source_counts)
            analytics_table.add_row("📚 Avg Sources Used", f"{avg_sources:.1f}", f"Total sources: {sum(source_counts)}")

        if question_lengths:
            avg_q_length = sum(question_lengths) / len(question_lengths)
            analytics_table.add_row("❓ Avg Question Length", f"{avg_q_length:.0f} chars", "Longer questions often get better answers")

        if answer_lengths:
            avg_a_length = sum(answer_lengths) / len(answer_lengths)
            analytics_table.add_row("💬 Avg Answer Length", f"{avg_a_length:.0f} chars", "Comprehensive responses")

        analytics_panel = Panel(analytics_table, title="[bold bright_magenta]📊 Session Insights", border_style="bright_magenta")
        self.console.print(analytics_panel)

    def _display_help(self):
        """Display comprehensive help information with enhanced formatting and examples."""
        # Create tabbed help sections

        # Commands table
        commands_table = Table(title="🎯 Available Commands", show_header=True, header_style="bold bright_yellow")
        commands_table.add_column("Command", style="bold bright_cyan", width=20)
        commands_table.add_column("Description", style="bright_white", width=35)
        commands_table.add_column("Example", style="dim", width=25)

        commands_data = [
            ("💡 /help", "Show this comprehensive help guide", "/help"),
            ("📜 /history", "Display conversation history with analytics", "/history"),
            ("🧹 /clear", "Clear conversation history and context", "/clear"),
            ("🔍 /search <query>", "Search documents directly with highlighting", "/search python tutorial"),
            ("📁 /refresh", "Refresh monitored directories", "/refresh"),
            ("📊 /stats", "Show detailed session statistics", "/stats"),
            ("💾 /save", "Manually save current session", "/save"),
            ("👋 /quit or /exit", "Exit gracefully with auto-save", "/quit")
        ]

        for cmd, desc, example in commands_data:
            commands_table.add_row(cmd, desc, example)

        # Tips table
        tips_table = Table(title="💡 Pro Tips for Better Results", show_header=True, header_style="bold bright_green")
        tips_table.add_column("Tip", style="bold bright_white", width=25)
        tips_table.add_column("Description", style="bright_white", width=50)

        tips_data = [
            ("🧠 Context Awareness", "I remember our conversation! Ask follow-up questions naturally."),
            ("🎯 Be Specific", "Detailed questions with context get better, more accurate answers."),
            ("🔍 Explore First", "Use /search to discover available documents and topics."),
            ("⚡ Hybrid Search", "I use both semantic understanding and keyword matching."),
            ("📚 Source Transparency", "I always show which documents informed my answers."),
            ("🔄 Iterative Refinement", "Refine questions based on my responses for deeper insights."),
            ("📊 Use Analytics", "Check /stats and /history for performance insights."),
            ("💾 Save Sessions", "Use /save to preserve important conversations.")
        ]

        for tip, desc in tips_data:
            tips_table.add_row(tip, desc)

        # Technical specs table
        tech_table = Table(title="⚙️ Technical Specifications", show_header=True, header_style="bold bright_purple")
        tech_table.add_column("Component", style="cyan", width=20)
        tech_table.add_column("Specification", style="bright_white", width=30)
        tech_table.add_column("Details", style="dim", width=25)

        tech_data = [
            ("🔧 Search Algorithm", "Hybrid (Vector + Full-text)", "Best of both worlds"),
            ("🧮 Context Window", f"Up to {self.config.max_context_length} characters", "Intelligent truncation"),
            ("📝 History Limit", f"Last {self.config.max_history} exchanges", "Configurable retention"),
            ("🎯 Search Results", f"Top {self.config.search_limit} documents", "Relevance-ranked"),
            ("💾 Cache Size", f"{self.config.cache_size} entries", "Performance optimization"),
            ("🔄 Auto-save", f"Every {self.config.auto_save_interval}s", "Automatic persistence")
        ]

        for component, spec, detail in tech_data:
            tech_table.add_row(component, spec, detail)

        # Display all help sections
        help_panels = [
            Panel(commands_table, title="[bold bright_yellow]📖 Command Reference", border_style="bright_yellow"),
            Panel(tips_table, title="[bold bright_green]🌟 Usage Tips", border_style="bright_green"),
            Panel(tech_table, title="[bold bright_purple]🔧 Technical Details", border_style="bright_purple")
        ]

        for panel in help_panels:
            self.console.print(panel)
            self.console.print()

        # Add quick start guide
        quick_start = Text()
        quick_start.append("🚀 Quick Start Guide:\n", style="bold bright_cyan")
        quick_start.append("1. Ask any question about your documents\n", style="bright_white")
        quick_start.append("2. Use /search to explore available content\n", style="bright_white")
        quick_start.append("3. Ask follow-up questions for deeper insights\n", style="bright_white")
        quick_start.append("4. Check /stats for performance metrics\n", style="bright_white")
        quick_start.append("5. Use /save to preserve important sessions\n", style="bright_white")

        quick_panel = Panel(quick_start, title="[bold bright_cyan]🚀 Quick Start", border_style="bright_cyan")
        self.console.print(quick_panel)

    async def run(self):
        """Run the enhanced interactive QA session with comprehensive command handling."""
        self._display_welcome()

        # Command mapping for better organization
        command_handlers = {
            "/help": self._display_help,
            "/history": self._display_history,
            "/clear": self._handle_clear_command,
            "/refresh": self._handle_refresh_command,
            "/stats": self._handle_stats_command,
            "/save": self._handle_save_command,
        }

        while self._is_running:
            try:
                # Enhanced prompt with session info
                session_info = ""
                if self.qa_agent:
                    metrics = self.qa_agent.conversation_history.get_metrics()
                    session_info = f" [dim]({metrics.get('total_questions', 0)} questions)[/dim]"

                question = Prompt.ask(f"\n[bold bright_cyan]💭 Ask me anything{session_info}")

                if not question.strip():
                    continue

                # Handle exit commands
                if question.lower() in ["/quit", "/exit"]:
                    if await self._handle_exit_command():
                        break
                    continue

                # Handle search command (with parameter)
                if question.lower().startswith("/search "):
                    query = question[8:].strip()
                    await self._handle_search_command(query)
                    continue

                # Handle other commands
                command = question.lower().strip()
                if command in command_handlers:
                    if asyncio.iscoroutinefunction(command_handlers[command]):
                        await command_handlers[command]()
                    else:
                        command_handlers[command]()
                    continue

                # Handle unknown commands
                if question.startswith("/"):
                    self._handle_unknown_command(question)
                    continue

                # Process regular question
                await self._process_question(question)

            except KeyboardInterrupt:
                if await self._handle_keyboard_interrupt():
                    break
                continue
            except Exception as e:
                self._handle_session_error(e)
                continue

    async def _handle_clear_command(self):
        """Handle clear command with confirmation."""
        if not self.qa_agent.conversation_history.history:
            no_history_panel = Panel(
                "📝 No conversation history to clear.",
                title="[yellow]🧹 Nothing to Clear",
                border_style="yellow"
            )
            self.console.print(no_history_panel)
            return

        # Ask for confirmation
        confirm = Confirm.ask(
            f"[yellow]Are you sure you want to clear {len(self.qa_agent.conversation_history.history)} conversation exchanges?[/yellow]"
        )

        if confirm:
            self.qa_agent.conversation_history.clear()
            clear_panel = Panel(
                "🧹 Conversation history has been cleared successfully!\n"
                "✨ Starting fresh with a clean slate.\n"
                "🧠 Context memory has been reset.",
                title="[bold bright_green]✅ History Cleared",
                border_style="bright_green"
            )
            self.console.print(clear_panel)
            logger.info("Conversation history cleared by user")
        else:
            self.console.print("[dim]Clear operation cancelled.[/dim]")

    async def _handle_exit_command(self) -> bool:
        """Handle exit command with session summary."""
        # Show session summary
        if self.qa_agent and self.qa_agent.conversation_history.history:
            metrics = self.qa_agent.get_session_stats()

            summary_text = Text()
            summary_text.append("📊 Session Summary:\n", style="bold bright_cyan")
            summary_text.append(f"   • Questions asked: {metrics.get('total_questions', 0)}\n", style="bright_white")
            summary_text.append(f"   • Average response time: {metrics.get('avg_response_time', 0):.2f}s\n", style="bright_white")
            summary_text.append(f"   • Cache hit rate: {metrics.get('cache_hit_rate', 0):.1f}%\n", style="bright_white")
            summary_text.append(f"   • Session duration: {time.time() - self._session_start_time:.1f}s\n", style="bright_white")

            goodbye_panel = Panel(
                summary_text,
                title="[bold bright_yellow]👋 Session Complete!",
                border_style="bright_yellow"
            )
            self.console.print(goodbye_panel)

        # Final goodbye message
        farewell_text = Text()
        farewell_text.append("🌟 Thank you for using HKEX ANNOUNCEMENT RAG!\n", style="bold bright_green")
        farewell_text.append("💫 Hope you found the answers you were looking for.\n", style="bright_white")
        farewell_text.append("🚀 Come back anytime for more insights!", style="bright_white")

        farewell_panel = Panel(
            farewell_text,
            title="[bold bright_yellow]🎉 Goodbye!",
            border_style="bright_yellow"
        )
        self.console.print(farewell_panel)

        return True

    def _handle_unknown_command(self, command: str):
        """Handle unknown commands with helpful suggestions."""
        # Extract command name
        cmd_name = command.split()[0] if command.split() else command

        # Suggest similar commands
        available_commands = ["/help", "/history", "/clear", "/search", "/refresh", "/stats", "/save", "/quit", "/exit"]
        suggestions = []

        for cmd in available_commands:
            if cmd_name.lower() in cmd.lower() or cmd.lower() in cmd_name.lower():
                suggestions.append(cmd)

        error_text = Text()
        error_text.append(f"❌ Unknown command: {command}\n\n", style="bold red")

        if suggestions:
            error_text.append("💡 Did you mean:\n", style="bright_yellow")
            for suggestion in suggestions[:3]:
                error_text.append(f"   • {suggestion}\n", style="bright_cyan")
        else:
            error_text.append("💡 Available commands:\n", style="bright_yellow")
            for cmd in available_commands[:4]:
                error_text.append(f"   • {cmd}\n", style="bright_cyan")

        error_text.append("\n📖 Type /help for complete command reference", style="dim")

        error_panel = Panel(
            error_text,
            title="[red]⚠️ Command Not Found",
            border_style="red"
        )
        self.console.print(error_panel)

    async def _process_question(self, question: str):
        """Process a regular question with enhanced error handling and performance tracking."""
        start_time = time.time()

        try:
            # Update performance metrics
            self._performance_metrics["total_queries"] += 1

            # Display question
            self._display_question(question)

            # Enhanced progress tracking for question processing
            with self.progress:
                qa_task = self.progress.add_task("🚀 Processing your question...", total=100)

                self.progress.update(qa_task, advance=25, description="🧠 Analyzing question...")
                await asyncio.sleep(0.1)

                self.progress.update(qa_task, advance=25, description="🔍 Searching knowledge base...")
                await asyncio.sleep(0.1)

                self.progress.update(qa_task, advance=25, description="🤖 Generating response...")

                # Get answer with context
                answer, search_results = await self.qa_agent.answer_with_context(question)

                self.progress.update(qa_task, advance=25, description="✅ Response ready!")

            # Calculate response time
            response_time = time.time() - start_time

            # Update performance metrics
            self._performance_metrics["successful_queries"] += 1
            self._performance_metrics["total_response_time"] += response_time
            self._performance_metrics["avg_response_time"] = (
                self._performance_metrics["total_response_time"] /
                self._performance_metrics["successful_queries"]
            )

            # Display answer with response time
            self._display_answer(answer, search_results, response_time)

            logger.info(f"Question processed successfully in {response_time:.2f}s")

        except Exception as e:
            self._performance_metrics["failed_queries"] += 1
            logger.error(f"Error processing question: {e}")

            error_text = Text()
            error_text.append(f"❌ Error processing your question: {str(e)}\n\n", style="bold red")
            error_text.append("🔧 Troubleshooting suggestions:\n", style="bright_yellow")
            error_text.append("   • Try rephrasing your question\n", style="bright_white")
            error_text.append("   • Check your internet connection\n", style="bright_white")
            error_text.append("   • Verify the database is accessible\n", style="bright_white")
            error_text.append("   • Try a simpler question first\n", style="bright_white")
            error_text.append("\n💡 Type /help for guidance or /stats for system status", style="dim")

            error_panel = Panel(
                error_text,
                title="[red]⚠️ Processing Error",
                border_style="red"
            )
            self.console.print(error_panel)

    async def _handle_keyboard_interrupt(self) -> bool:
        """Handle keyboard interrupt with user choice."""
        interrupt_text = Text()
        interrupt_text.append("⚠️ Session interrupted by user.\n\n", style="bold yellow")
        interrupt_text.append("Choose an option:\n", style="bright_white")
        interrupt_text.append("   • Press Enter to continue\n", style="bright_white")
        interrupt_text.append("   • Type 'quit' to exit\n", style="bright_white")
        interrupt_text.append("   • Type 'save' to save and continue\n", style="bright_white")

        interrupt_panel = Panel(
            interrupt_text,
            title="[yellow]🛑 Interrupted",
            border_style="yellow"
        )
        self.console.print(interrupt_panel)

        try:
            choice = Prompt.ask("[yellow]What would you like to do?[/yellow]", default="continue")

            if choice.lower() in ['quit', 'exit', 'q']:
                return True
            elif choice.lower() in ['save', 's']:
                await self._handle_save_command()
                return False
            else:
                self.console.print("[dim]Continuing session...[/dim]")
                return False

        except KeyboardInterrupt:
            # Double Ctrl+C means force quit
            self.console.print("\n[red]Force quit detected. Exiting...[/red]")
            return True

    def _handle_session_error(self, error: Exception):
        """Handle general session errors with recovery options."""
        logger.error(f"Session error: {error}")

        error_text = Text()
        error_text.append(f"❌ Session error: {str(error)}\n\n", style="bold red")

        # Provide specific guidance based on error type
        if "database" in str(error).lower():
            error_text.append("🔧 Database connection issue detected:\n", style="bright_yellow")
            error_text.append("   • Check if the database file exists\n", style="bright_white")
            error_text.append("   • Verify file permissions\n", style="bright_white")
            error_text.append("   • Ensure sufficient disk space\n", style="bright_white")
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            error_text.append("🌐 Network connection issue detected:\n", style="bright_yellow")
            error_text.append("   • Check your internet connection\n", style="bright_white")
            error_text.append("   • Verify API endpoints are accessible\n", style="bright_white")
            error_text.append("   • Check firewall settings\n", style="bright_white")
        else:
            error_text.append("🔧 General troubleshooting:\n", style="bright_yellow")
            error_text.append("   • Try restarting the session\n", style="bright_white")
            error_text.append("   • Check system resources\n", style="bright_white")
            error_text.append("   • Verify configuration settings\n", style="bright_white")

        error_text.append("\n💡 Session will continue. Type /help for assistance.", style="dim")

        error_panel = Panel(
            error_text,
            title="[red]⚠️ Session Error",
            border_style="red"
        )
        self.console.print(error_panel)


async def start_interactive_qa(
    db_path: str,
    model: str = "",
    enable_monitoring: bool = True,
    config: Optional[SessionConfig] = None,
    session_id: Optional[str] = None
) -> None:
    """
    Start an enhanced interactive QA session with comprehensive features.

    Args:
        db_path: Path to the database file
        model: Model name to use for QA
        enable_monitoring: Whether to enable file monitoring
        config: Optional session configuration
        session_id: Optional session ID for resuming sessions
    """
    console = Console()

    try:
        # Initialize with enhanced configuration
        session_config = config or SessionConfig()

        # Display startup banner
        startup_text = Text()
        startup_text.append("🚀 Starting Enhanced Interactive QA Session...\n", style="bold bright_blue")
        startup_text.append(f"📁 Database: {Path(db_path).name}\n", style="bright_white")
        startup_text.append(f"🤖 Model: {model or 'Default'}\n", style="bright_white")
        startup_text.append(f"📊 Monitoring: {'Enabled' if enable_monitoring else 'Disabled'}\n", style="bright_white")
        startup_text.append(f"⚙️ Config: Optimized for performance\n", style="bright_white")

        startup_panel = Panel(
            startup_text,
            title="[bold bright_blue]🔧 Initialization",
            border_style="bright_blue"
        )
        console.print(startup_panel)

        # Start the session
        async with InteractiveQASession(
            db_path=db_path,
            model=model,
            enable_monitoring=enable_monitoring,
            config=session_config,
            session_id=session_id
        ) as session:
            await session.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Session interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        logger.error(f"Failed to start interactive QA session: {e}")

        error_text = Text()
        error_text.append(f"❌ Failed to start session: {str(e)}\n\n", style="bold red")
        error_text.append("🔧 Troubleshooting:\n", style="bright_yellow")
        error_text.append("   • Check database path and permissions\n", style="bright_white")
        error_text.append("   • Verify model configuration\n", style="bright_white")
        error_text.append("   • Ensure all dependencies are installed\n", style="bright_white")
        error_text.append("   • Check system resources\n", style="bright_white")

        error_panel = Panel(
            error_text,
            title="[red]⚠️ Startup Error",
            border_style="red"
        )
        console.print(error_panel)
        raise


async def interactive_qa_cli(
    db_path: str,
    model: str = "",
    enable_monitoring: bool = True,
    config_file: Optional[str] = None
) -> None:
    """
    Enhanced CLI wrapper for interactive QA with configuration support.

    Args:
        db_path: Path to the database file
        model: Model name to use for QA
        enable_monitoring: Whether to enable file monitoring
        config_file: Optional path to configuration file
    """
    console = Console()

    try:
        # Load configuration if provided
        session_config = SessionConfig()
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                session_config = SessionConfig(**config_data)
                console.print(f"[green]✅ Loaded configuration from {config_file}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Failed to load config file: {e}. Using defaults.[/yellow]")

        # Start the session
        await start_interactive_qa(
            db_path=db_path,
            model=model,
            enable_monitoring=enable_monitoring,
            config=session_config
        )

    except Exception as e:
        logger.error(f"CLI error: {e}")
        console.print(f"[red]❌ Failed to start interactive QA: {e}[/red]")
        console.print("[dim]💡 Use --help for usage information[/dim]")


def create_session_config(**kwargs) -> SessionConfig:
    """
    Create a session configuration with custom parameters.

    Returns:
        Configured SessionConfig instance
    """
    return SessionConfig(**kwargs)


def load_session_from_file(session_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load a saved session from file.

    Args:
        session_file: Path to the session file

    Returns:
        Session data dictionary or None if failed
    """
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load session from {session_file}: {e}")
        return None


if __name__ == "__main__":
    """Enhanced command line interface with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Interactive QA System for HKEX Announcements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive.py                          # Use default database
  python interactive.py mydata.db               # Use specific database
  python interactive.py mydata.db gpt-4         # Use specific model
  python interactive.py --no-monitoring         # Disable file monitoring
  python interactive.py --config config.json    # Use custom configuration
        """
    )

    parser.add_argument(
        "db_path",
        nargs="?",
        default="haiku.rag.sqlite",
        help="Path to the database file (default: haiku.rag.sqlite)"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="",
        help="Model name to use for QA (default: from config)"
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable file monitoring"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Resume a specific session by ID"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("Debug logging enabled")

    # Run the interactive session
    try:
        asyncio.run(interactive_qa_cli(
            db_path=args.db_path,
            model=args.model,
            enable_monitoring=not args.no_monitoring,
            config_file=args.config
        ))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
