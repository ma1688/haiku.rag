#!/usr/bin/env python3
"""
美化界面演示脚本 - Beautiful Interactive QA Demo

这个脚本展示了美化后的交互式QA界面的所有功能。
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haiku.rag.qa.interactive import start_interactive_qa
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def display_demo_info():
    """显示演示信息"""
    console = Console()
    
    demo_text = Text()
    demo_text.append("🎨 ", style="bold bright_blue")
    demo_text.append("Beautiful Interactive QA Demo", style="bold bright_white")
    demo_text.append("\n\n")
    demo_text.append("✨ Features showcased in this demo:\n", style="bold bright_yellow")
    demo_text.append("  🎯 Enhanced welcome screen with gradient-like styling\n", style="bright_white")
    demo_text.append("  💬 Beautiful question and answer panels\n", style="bright_white")
    demo_text.append("  🔍 Improved search results with relevance indicators\n", style="bright_white")
    demo_text.append("  📜 Elegant conversation history display\n", style="bright_white")
    demo_text.append("  💡 Comprehensive help system\n", style="bright_white")
    demo_text.append("  🎨 Rich color coding and emoji indicators\n", style="bright_white")
    demo_text.append("  ⚡ Enhanced error handling and user feedback\n", style="bright_white")
    demo_text.append("\n")
    demo_text.append("🚀 Ready to start the beautiful interactive experience!\n", style="bold bright_green")
    demo_text.append("📝 Try these commands once inside:\n", style="dim")
    demo_text.append("   • /help - See the beautiful help system\n", style="dim")
    demo_text.append("   • /search <query> - Try the enhanced search\n", style="dim")
    demo_text.append("   • Ask any question to see the improved Q&A flow\n", style="dim")
    
    panel = Panel(
        demo_text,
        title="[bold bright_blue]🎨 Beautiful Interface Demo",
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


async def main():
    """主函数"""
    # 显示演示信息
    display_demo_info()
    
    # 检查数据库文件
    db_path = "haiku.rag.sqlite"
    if not Path(db_path).exists():
        console = Console()
        error_panel = Panel(
            f"❌ Database file not found: {db_path}\n"
            "💡 Please ensure you have a RAG database file in the current directory.\n"
            "🔧 You can create one using the haiku-rag CLI tools.",
            title="[red]⚠️ Database Not Found",
            border_style="red"
        )
        console.print(error_panel)
        return
    
    # 启动美化的交互式QA会话
    try:
        await start_interactive_qa(db_path)
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]👋 Demo interrupted. Thanks for trying the beautiful interface![/yellow]")
    except Exception as e:
        console = Console()
        error_panel = Panel(
            f"❌ Error starting demo: {str(e)}\n"
            "🔧 Please check your setup and try again.",
            title="[red]⚠️ Demo Error",
            border_style="red"
        )
        console.print(error_panel)


if __name__ == "__main__":
    asyncio.run(main())
