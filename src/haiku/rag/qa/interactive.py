"""
Interactive QA agent that supports conversational loops with context preservation.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from haiku.rag.client import HaikuRAG
from haiku.rag.qa import get_qa_agent
from haiku.rag.qa.base import QuestionAnswerAgentBase


class ConversationHistory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
    
    def add_exchange(self, question: str, answer: str, search_results: Optional[List] = None):
        """Add a question-answer exchange to history."""
        exchange = {
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "search_results": search_results or []
        }
        self.history.append(exchange)
        
        # Keep only the most recent exchanges
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_summary(self) -> str:
        """Generate a context summary from recent conversation history."""
        if not self.history:
            return ""
        
        context_parts = []
        for exchange in self.history[-3:]:  # Last 3 exchanges for context
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"A: {exchange['answer'][:200]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()
        self.session_start = datetime.now()


class ContextAwareQAAgent(QuestionAnswerAgentBase):
    """QA Agent that maintains conversation context."""
    
    def __init__(self, client: HaikuRAG, model: str = "", max_context_length: int = 2000):
        super().__init__(client, model)
        self.base_agent = get_qa_agent(client, model)
        self.conversation_history = ConversationHistory()
        self.max_context_length = max_context_length
    
    async def answer_with_context(self, question: str) -> tuple[str, List]:
        """Answer a question with conversation context."""
        # Get conversation context
        context = self.conversation_history.get_context_summary()
        
        # Enhance question with context if available
        enhanced_question = question
        if context and len(context) < self.max_context_length:
            enhanced_question = f"""
Previous conversation context:
{context}

Current question: {question}

Please answer the current question, taking into account the conversation context if relevant.
"""
        
        # Get search results for transparency
        search_results = await self._client.search(question, limit=5)
        
        # Get answer from base agent
        answer = await self.base_agent.answer(enhanced_question)
        
        # Add to conversation history
        self.conversation_history.add_exchange(question, answer, search_results)
        
        return answer, search_results
    
    async def answer(self, question: str) -> str:
        """Standard answer method for compatibility."""
        answer, _ = await self.answer_with_context(question)
        return answer


class InteractiveQASession:
    """Interactive QA session with rich console interface."""
    
    def __init__(self, db_path: str, model: str = ""):
        self.db_path = db_path
        self.model = model
        self.console = Console()
        self.client: Optional[HaikuRAG] = None
        self.qa_agent: Optional[ContextAwareQAAgent] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = HaikuRAG(self.db_path)
        await self.client.__aenter__()
        self.qa_agent = ContextAwareQAAgent(self.client, self.model)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _display_welcome(self):
        """Display enhanced welcome message with beautiful styling."""
        # Create header with gradient-like effect
        header = Text()
        header.append("╭", style="bold bright_blue")
        header.append("─" * 60, style="bright_blue")
        header.append(" 🚀 Interactive QA Session ", style="bold bright_white on bright_blue")
        header.append("─" * 60, style="bright_blue")
        header.append("╮", style="bold bright_blue")

        # Main content
        content = Text()
        content.append("\n")

        # Title section with enhanced styling
        content.append("    🤖 ", style="bold bright_blue")
        content.append("HKEX ANNOUNCEMENT RAG", style="bold bright_white")
        content.append(" Interactive QA\n", style="bold bright_blue")

        # Author and version info with better formatting
        content.append("    ", style="dim")
        content.append("👨‍💻 Author: ", style="dim")
        content.append("MAXJ", style="bold bright_green")
        content.append("    📦 Version: ", style="dim")
        content.append("v0.1", style="bold bright_yellow")
        content.append("    🕒 ", style="dim")
        content.append(datetime.now().strftime("%Y-%m-%d %H:%M"), style="dim")
        content.append("\n\n")

        # Description with better formatting
        content.append("    💬 ", style="bright_blue")
        content.append("Type your questions and I'll search the knowledge base to provide intelligent answers.\n", style="bright_white")
        content.append("    🧠 ", style="bright_purple")
        content.append("I maintain conversation context and can handle follow-up questions.\n\n", style="bright_white")

        # Commands section with enhanced styling
        content.append("    ⚡ ", style="bright_yellow")
        content.append("Special Commands:", style="bold bright_white")
        content.append("\n")

        commands = [
            ("💡 /help", "Show detailed help and tips"),
            ("📜 /history", "Display conversation history"),
            ("🧹 /clear", "Clear conversation history"),
            ("🔍 /search <query>", "Search documents directly"),
            ("👋 /quit or /exit", "Exit the session gracefully")
        ]

        for emoji_cmd, description in commands:
            content.append("      ", style="dim")
            content.append("• ", style="bright_blue")
            content.append(emoji_cmd, style="bold bright_cyan")
            content.append(" - ", style="dim")
            content.append(description, style="bright_white")
            content.append("\n")

        # Footer
        footer = Text()
        footer.append("╰", style="bold bright_blue")
        footer.append("─" * 140, style="bright_blue")
        footer.append("╯", style="bold bright_blue")

        # Display everything
        self.console.print()
        self.console.print(header)
        self.console.print(content)
        self.console.print(footer)
        self.console.print()

        # Add a motivational message
        motivation = Text()
        motivation.append("✨ ", style="bright_yellow")
        motivation.append("Ready to explore your knowledge base! Ask me anything...", style="italic bright_white")
        self.console.print(motivation)
        self.console.print()
    
    def _display_question(self, question: str):
        """Display user question with enhanced styling."""
        # Create a beautiful question display
        question_panel = Panel(
            Text(question, style="bright_white"),
            title="[bold bright_blue]👤 Your Question",
            title_align="left",
            border_style="bright_blue",
            padding=(0, 1)
        )
        self.console.print(question_panel)
        self.console.print()

    def _display_answer(self, answer: str, search_results: List = None):
        """Display AI answer with enhanced styling and search context."""
        # Display answer in a beautiful panel
        answer_panel = Panel(
            Markdown(answer),
            title="[bold bright_green]🤖 AI Assistant",
            title_align="left",
            border_style="bright_green",
            padding=(0, 1)
        )
        self.console.print(answer_panel)

        # Display search results if available with enhanced styling
        if search_results:
            self.console.print()
            sources_text = Text()
            sources_text.append("📚 Knowledge Sources Used:\n", style="bold bright_yellow")

            for i, (chunk, score) in enumerate(search_results[:3], 1):
                # Score indicator with color coding
                if score > 0.8:
                    score_style = "bold bright_green"
                    score_icon = "🟢"
                elif score > 0.6:
                    score_style = "bold bright_yellow"
                    score_icon = "🟡"
                else:
                    score_style = "bold bright_red"
                    score_icon = "🔴"

                sources_text.append(f"\n  {score_icon} ", style=score_style)
                sources_text.append(f"Source {i}", style="bold bright_cyan")
                sources_text.append(f" (Relevance: {score:.1%})", style=score_style)

                if chunk.document_uri:
                    sources_text.append(f"\n    📄 Document: ", style="dim")
                    sources_text.append(f"{chunk.document_uri}", style="bright_blue")

                # Preview of content
                preview = chunk.content[:150].replace('\n', ' ').strip()
                if len(chunk.content) > 150:
                    preview += "..."
                sources_text.append(f"\n    💭 Preview: ", style="dim")
                sources_text.append(f"{preview}", style="bright_white")
                sources_text.append("\n")

            sources_panel = Panel(
                sources_text,
                title="[dim]📖 Reference Materials",
                title_align="left",
                border_style="dim",
                padding=(0, 1)
            )
            self.console.print(sources_panel)

        self.console.print()

    async def _handle_search_command(self, query: str):
        """Handle direct search command with enhanced display."""
        if not query.strip():
            error_panel = Panel(
                "[red]❌ Please provide a search query after /search command.[/red]",
                title="[red]Error",
                border_style="red"
            )
            self.console.print(error_panel)
            return

        # Show search progress
        search_panel = Panel(
            f"🔍 Searching knowledge base for: [bold bright_yellow]{query}[/bold bright_yellow]",
            title="[bright_blue]🔎 Search in Progress",
            border_style="bright_blue"
        )
        self.console.print(search_panel)

        with self.console.status("[bold bright_blue]🔍 Searching documents...", spinner="dots"):
            results = await self.client.search(query, limit=5)

        if not results:
            no_results_panel = Panel(
                "🚫 No matching documents found. Try different keywords or check your query.",
                title="[yellow]No Results",
                border_style="yellow"
            )
            self.console.print(no_results_panel)
            return

        # Display results with enhanced formatting
        results_text = Text()
        results_text.append(f"✅ Found {len(results)} relevant documents:\n\n", style="bold bright_green")

        for i, (chunk, score) in enumerate(results, 1):
            # Score visualization
            if score > 0.8:
                score_style = "bold bright_green"
                score_bar = "█████"
            elif score > 0.6:
                score_style = "bold bright_yellow"
                score_bar = "████░"
            elif score > 0.4:
                score_style = "bold bright_orange"
                score_bar = "███░░"
            else:
                score_style = "bold bright_red"
                score_bar = "██░░░"

            results_text.append(f"📄 Result {i}", style="bold bright_cyan")
            results_text.append(f" │ Relevance: {score:.1%} ", style=score_style)
            results_text.append(f"[{score_bar}]", style=score_style)
            results_text.append("\n")

            if chunk.document_uri:
                results_text.append(f"   📁 Source: ", style="dim")
                results_text.append(f"{chunk.document_uri}", style="bright_blue")
                results_text.append("\n")

            # Content preview with better formatting
            content_preview = chunk.content[:250].replace('\n', ' ').strip()
            if len(chunk.content) > 250:
                content_preview += "..."

            results_text.append(f"   💭 Content: ", style="dim")
            results_text.append(f"{content_preview}", style="bright_white")
            results_text.append("\n\n")

        results_panel = Panel(
            results_text,
            title="[bright_green]🔍 Search Results",
            border_style="bright_green",
            padding=(0, 1)
        )
        self.console.print(results_panel)

    def _display_history(self):
        """Display conversation history with enhanced formatting."""
        if not self.qa_agent.conversation_history.history:
            no_history_panel = Panel(
                "📝 No conversation history yet. Start asking questions to build your session history!",
                title="[yellow]📜 Conversation History",
                border_style="yellow"
            )
            self.console.print(no_history_panel)
            return

        # Create history content
        history_text = Text()
        session_duration = datetime.now() - self.qa_agent.conversation_history.session_start
        history_text.append(f"🕒 Session Duration: {str(session_duration).split('.')[0]}\n", style="dim")
        history_text.append(f"💬 Total Exchanges: {len(self.qa_agent.conversation_history.history)}\n\n", style="dim")

        for i, exchange in enumerate(self.qa_agent.conversation_history.history, 1):
            timestamp = exchange["timestamp"].strftime("%H:%M:%S")

            # Exchange header
            history_text.append(f"┌─ Exchange {i} ", style="bright_blue")
            history_text.append(f"[{timestamp}]", style="dim")
            history_text.append(" ─" * 50, style="bright_blue")
            history_text.append("\n")

            # Question
            history_text.append("│ ", style="bright_blue")
            history_text.append("👤 Question: ", style="bold bright_blue")
            history_text.append(f"{exchange['question']}", style="bright_white")
            history_text.append("\n│\n", style="bright_blue")

            # Answer preview
            answer_preview = exchange['answer'][:200].replace('\n', ' ').strip()
            if len(exchange['answer']) > 200:
                answer_preview += "..."

            history_text.append("│ ", style="bright_blue")
            history_text.append("🤖 Answer: ", style="bold bright_green")
            history_text.append(f"{answer_preview}", style="bright_white")
            history_text.append("\n")

            # Sources count if available
            if exchange.get('search_results'):
                source_count = len(exchange['search_results'])
                history_text.append("│ ", style="bright_blue")
                history_text.append(f"📚 Sources: {source_count} documents referenced", style="dim")
                history_text.append("\n")

            history_text.append("└" + "─" * 70, style="bright_blue")
            history_text.append("\n\n")

        history_panel = Panel(
            history_text,
            title="[bold bright_blue]📜 Conversation History",
            border_style="bright_blue",
            padding=(0, 1)
        )
        self.console.print(history_panel)

    def _display_help(self):
        """Display comprehensive help information with beautiful formatting."""
        help_content = Text()

        # Commands section
        help_content.append("🎯 Available Commands:\n\n", style="bold bright_yellow")

        commands_help = [
            ("💡 /help", "Show this comprehensive help guide", "Get detailed information about all features"),
            ("📜 /history", "Display conversation history", "View all your questions and answers from this session"),
            ("🧹 /clear", "Clear conversation history", "Start fresh - removes all context and history"),
            ("🔍 /search <query>", "Search documents directly", "Find specific information without asking a question"),
            ("👋 /quit or /exit", "Exit the session gracefully", "Save your progress and close the application")
        ]

        for emoji_cmd, short_desc, long_desc in commands_help:
            help_content.append("  ▶ ", style="bright_blue")
            help_content.append(emoji_cmd, style="bold bright_cyan")
            help_content.append(f" - {short_desc}", style="bright_white")
            help_content.append(f"\n    💭 {long_desc}", style="dim")
            help_content.append("\n\n")

        # Tips section
        help_content.append("💡 Pro Tips for Better Results:\n\n", style="bold bright_green")

        tips = [
            ("🧠 Context Awareness", "I remember our conversation! Ask follow-up questions naturally."),
            ("🎯 Be Specific", "Detailed questions get better answers. Include context and specifics."),
            ("🔍 Explore First", "Use /search to discover what documents are available."),
            ("⚡ Hybrid Search", "I use both semantic and keyword search for comprehensive results."),
            ("📚 Source Transparency", "I always show you which documents I used for my answers."),
            ("🔄 Iterative Queries", "Refine your questions based on my responses for deeper insights.")
        ]

        for tip_title, tip_desc in tips:
            help_content.append("  ✨ ", style="bright_yellow")
            help_content.append(tip_title, style="bold bright_white")
            help_content.append(f"\n    {tip_desc}", style="bright_white")
            help_content.append("\n\n")

        # Technical info
        help_content.append("⚙️ Technical Information:\n\n", style="bold bright_purple")
        help_content.append("  🔧 Search Algorithm: Hybrid (Vector + Full-text)\n", style="bright_white")
        help_content.append("  🧮 Context Window: Up to 2000 characters\n", style="bright_white")
        help_content.append("  📝 History Limit: Last 10 exchanges\n", style="bright_white")
        help_content.append("  🎯 Source Limit: Top 5 most relevant documents\n", style="bright_white")

        help_panel = Panel(
            help_content,
            title="[bold bright_yellow]📖 Comprehensive Help Guide",
            border_style="bright_yellow",
            padding=(1, 2)
        )
        self.console.print(help_panel)

    async def run(self):
        """Run the interactive QA session."""
        self._display_welcome()

        while True:
            try:
                # Get user input with enhanced prompt
                question = Prompt.ask("\n[bold bright_cyan]💭 Ask me anything")

                if not question.strip():
                    continue

                # Handle special commands
                if question.lower() in ["/quit", "/exit"]:
                    goodbye_panel = Panel(
                        "👋 Thank you for using HKEX ANNOUNCEMENT RAG!\n🌟 Hope you found the answers you were looking for.\n💫 Come back anytime for more insights!",
                        title="[bold bright_yellow]🎉 Goodbye!",
                        border_style="bright_yellow",
                        padding=(1, 2)
                    )
                    self.console.print(goodbye_panel)
                    break
                elif question.lower() == "/help":
                    self._display_help()
                    continue
                elif question.lower() == "/history":
                    self._display_history()
                    continue
                elif question.lower() == "/clear":
                    self.qa_agent.conversation_history.clear()
                    clear_panel = Panel(
                        "🧹 Conversation history has been cleared successfully!\n✨ Starting fresh with a clean slate.",
                        title="[bold bright_green]✅ History Cleared",
                        border_style="bright_green"
                    )
                    self.console.print(clear_panel)
                    continue
                elif question.lower().startswith("/search "):
                    query = question[8:].strip()  # Remove "/search " prefix
                    await self._handle_search_command(query)
                    continue
                elif question.startswith("/"):
                    error_panel = Panel(
                        f"❌ Unknown command: [bold]{question}[/bold]\n💡 Type [bold bright_cyan]/help[/bold bright_cyan] to see all available commands.",
                        title="[red]⚠️ Command Not Found",
                        border_style="red"
                    )
                    self.console.print(error_panel)
                    continue

                # Display question
                self._display_question(question)

                # Show enhanced thinking indicator
                thinking_messages = [
                    "🧠 Analyzing your question...",
                    "🔍 Searching knowledge base...",
                    "🤔 Processing information...",
                    "💭 Generating response..."
                ]

                with self.console.status("[bold bright_green]🚀 Working on your answer...", spinner="dots"):
                    answer, search_results = await self.qa_agent.answer_with_context(question)

                # Display answer
                self._display_answer(answer, search_results)

            except KeyboardInterrupt:
                interrupt_panel = Panel(
                    "⚠️ Session interrupted by user.\n💡 Type [bold bright_cyan]/quit[/bold bright_cyan] to exit properly or continue asking questions.",
                    title="[yellow]🛑 Interrupted",
                    border_style="yellow"
                )
                self.console.print(interrupt_panel)
                continue
            except Exception as e:
                error_panel = Panel(
                    f"❌ An error occurred: [bold red]{str(e)}[/bold red]\n🔧 Please try again or contact support if the issue persists.",
                    title="[red]⚠️ Error",
                    border_style="red"
                )
                self.console.print(error_panel)
                continue


async def start_interactive_qa(db_path: str, model: str = ""):
    """Start an interactive QA session."""
    async with InteractiveQASession(db_path, model) as session:
        await session.run()


# CLI integration function
async def interactive_qa_cli(db_path: str, model: str = ""):
    """CLI wrapper for interactive QA."""
    try:
        await start_interactive_qa(db_path, model)
    except Exception as e:
        console = Console()
        console.print(f"[red]Failed to start interactive QA: {e}[/red]")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Simple command line interface for testing
    db_path = sys.argv[1] if len(sys.argv) > 1 else "haiku.rag.sqlite"
    model = sys.argv[2] if len(sys.argv) > 2 else ""

    asyncio.run(start_interactive_qa(db_path, model))
