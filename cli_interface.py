#!/usr/bin/env python3
"""
Enhanced CLI Interface for Recallr
Provides a modern, user-friendly command-line interface with slash commands,
better visual feedback, and improved user experience.
"""

import os
import sys
import time
import shutil
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path


# Rich library for beautiful CLI interface
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich.padding import Padding
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console class
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def input(self, prompt=""):
            return input(prompt)

console = Console()

class SlashCommand:
    """Base class for slash commands"""
    def __init__(self, name: str, description: str, aliases: List[str] = None):
        self.name = name
        self.description = description
        self.aliases = aliases or []

    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        """Execute the command. Return True to continue main loop, False to exit."""
        raise NotImplementedError

class SummaryCommand(SlashCommand):
    """Generate conversation summary"""
    def __init__(self):
        super().__init__(
            name="summary",
            description="Generate a summary of the current conversation",
            aliases=["sum", "s"]
        )

    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        conversation_history = context.get('conversation_history', [])

        if not conversation_history:
            if RICH_AVAILABLE:
                console.print("📝 [yellow]No conversation history yet.[/yellow]")
            else:
                print("📝 No conversation history yet.")
            return True

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("🤖 Generating conversation summary...", total=None)
                try:
                    # Call the summary function from the main conversation
                    if 'summary_function' in context:
                        await context['summary_function']()
                    else:
                        console.print("❌ [red]Summary function not available[/red]")
                except Exception as e:
                    console.print(f"❌ [red]Error generating summary: {e}[/red]")
        else:
            print("🤖 Generating conversation summary...")
            try:
                if 'summary_function' in context:
                    await context['summary_function']()
                else:
                    print("❌ Summary function not available")
            except Exception as e:
                print(f"❌ Error generating summary: {e}")

        return True

class ClearCommand(SlashCommand):
    """Clear conversation history"""
    def __init__(self):
        super().__init__(
            name="clear",
            description="Clear the conversation history",
            aliases=["c", "reset", "new"]
        )

    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        if RICH_AVAILABLE:
            confirm = Confirm.ask("🧹 Are you sure you want to clear the conversation history?")
        else:
            response = input("🧹 Are you sure you want to clear the conversation history? (y/n): ")
            confirm = response.lower() in ['y', 'yes']

        if confirm:
            try:
                if 'clear_function' in context:
                    context['clear_function']()
                    if RICH_AVAILABLE:
                        console.print("✅ [green]Conversation history cleared![/green]")
                    else:
                        print("✅ Conversation history cleared!")
                else:
                    if RICH_AVAILABLE:
                        console.print("❌ [red]Clear function not available[/red]")
                    else:
                        print("❌ Clear function not available")
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"❌ [red]Error clearing history: {e}[/red]")
                else:
                    print(f"❌ Error clearing history: {e}")
        else:
            if RICH_AVAILABLE:
                console.print("📝 [yellow]Clear operation cancelled[/yellow]")
            else:
                print("📝 Clear operation cancelled")

        return True

class HelpCommand(SlashCommand):
    """Show help information"""
    def __init__(self):
        super().__init__(
            name="help",
            description="Show available commands and help information",
            aliases=["h", "?"]
        )

    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        commands = context.get('commands', {})

        if RICH_AVAILABLE:
            # Create a beautiful help table
            table = Table(title="🤖 Recallr Commands", box=box.ROUNDED)
            table.add_column("Command", style="cyan", no_wrap=True)
            table.add_column("Aliases", style="magenta")
            table.add_column("Description", style="white")

            for cmd_name, cmd in commands.items():
                aliases = ", ".join(f"/{alias}" for alias in cmd.aliases) if cmd.aliases else ""
                table.add_row(f"/{cmd_name}", aliases, cmd.description)

            # Add regular commands
            table.add_row("quit/exit/q", "", "Exit the application")
            table.add_row("refresh/reload", "", "Refresh academic knowledge base")

            console.print(table)
            console.print("\n💡 [bold cyan]Tips:[/bold cyan]")
            console.print("• Type your questions naturally - Recallr will route them to the best AI assistant")
            console.print("• Use [cyan]/summary[/cyan] to get an overview of your conversation")
            console.print("• Use [cyan]/clear[/cyan] to start fresh")
            console.print("• Recallr supports math, coding, and academic questions")
        else:
            print("\n🤖 Recallr Commands")
            print("=" * 50)
            for cmd_name, cmd in commands.items():
                aliases = ", ".join(f"/{alias}" for alias in cmd.aliases) if cmd.aliases else ""
                print(f"/{cmd_name:<12} {aliases:<20} {cmd.description}")
            print(f"{'quit/exit/q':<12} {'':<20} Exit the application")
            print(f"{'refresh':<12} {'':<20} Refresh academic knowledge base")
            print("\n💡 Tips:")
            print("• Type your questions naturally")
            print("• Use /summary to get conversation overview")
            print("• Use /clear to start fresh")

        return True

class StatusCommand(SlashCommand):
    """Show system status"""
    def __init__(self):
        super().__init__(
            name="status",
            description="Show system status and statistics",
            aliases=["info", "stats"]
        )

    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        conversation_history = context.get('conversation_history', [])

        if RICH_AVAILABLE:
            # Create status panel
            status_text = Text()
            status_text.append("📊 System Status\n\n", style="bold cyan")
            status_text.append(f"💬 Conversation Exchanges: {len(conversation_history) // 2}\n")
            status_text.append(f"📅 Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            status_text.append(f"🏠 Working Directory: {Path.cwd()}\n")
            status_text.append(f"📂 Data Directory: {'✅ Found' if Path('data').exists() else '❌ Missing'}\n")
            status_text.append(f"💾 Storage Directory: {'✅ Found' if Path('storage').exists() else '❌ Missing'}\n")

            # Environment variables status
            status_text.append("\n🔑 API Keys Status:\n")
            api_keys = {
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
                "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY"),
                "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
            }

            for key, value in api_keys.items():
                status = "✅ Set" if value else "❌ Missing"
                status_text.append(f"   {key}: {status}\n")

            panel = Panel(status_text, title="System Information", border_style="green")
            console.print(panel)
        else:
            print("\n📊 System Status")
            print("=" * 40)
            print(f"💬 Conversation Exchanges: {len(conversation_history) // 2}")
            print(f"📅 Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🏠 Working Directory: {Path.cwd()}")
            print(f"📂 Data Directory: {'✅ Found' if Path('data').exists() else '❌ Missing'}")
            print(f"💾 Storage Directory: {'✅ Found' if Path('storage').exists() else '❌ Missing'}")
            print("\n🔑 API Keys Status:")
            api_keys = {
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
                "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY"),
                "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
            }
            for key, value in api_keys.items():
                status = "✅ Set" if value else "❌ Missing"
                print(f"   {key}: {status}")

        return True

class ExitCommand(SlashCommand):
    """Exit the application"""
    def __init__(self):
        super().__init__(
            name="exit",
            description="Exit the application",
            aliases=["quit", "q", "bye"]
        )

    async def execute(self, args: List[str], context: Dict[str, Any]) -> bool:
        if RICH_AVAILABLE:
            console.print("👋 [bold green]Goodbye! Thanks for using Recallr![/bold green]")
        else:
            print("👋 Goodbye! Thanks for using Recallr!")
        return False

class EnhancedCLI:
    """Enhanced CLI interface with rich formatting and slash commands"""

    def __init__(self):
        self.commands: Dict[str, SlashCommand] = {}
        self.context: Dict[str, Any] = {
            'conversation_history': [],
            'session_start': datetime.now(),
            'commands': self.commands
        }

        # Register built-in commands
        self._register_commands()

        # Terminal width for formatting
        self.terminal_width = shutil.get_terminal_size().columns

    def _register_commands(self):
        """Register all available slash commands"""
        commands = [
            SummaryCommand(),
            ClearCommand(),
            HelpCommand(),
            StatusCommand(),
            ExitCommand(),
        ]

        for cmd in commands:
            self.register_command(cmd)

    def register_command(self, command: SlashCommand):
        """Register a new slash command"""
        self.commands[command.name] = command
        # Also register aliases
        for alias in command.aliases:
            self.commands[alias] = command

    def set_context_function(self, key: str, func: Callable):
        """Set context functions (e.g., summary_function, clear_function)"""
        self.context[key] = func

    def update_conversation_history(self, history: List[Dict[str, Any]]):
        """Update conversation history in context"""
        self.context['conversation_history'] = history

    def display_welcome(self):
        """Display welcome message with enhanced formatting"""
        if RICH_AVAILABLE:
            # Create ASCII art panel
            art = """
██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗     ██████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║     ██╔══██╗
██████╔╝█████╗  ██║     ███████║██║     ██║     ██████╔╝
██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║     ██╔══██╗
██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
            """

            welcome_panel = Panel(
                Align.center(art + "\n🤖 Your AI-Powered Learning Assistant 🤖"),
                border_style="bright_blue",
                title="Welcome to Recallr",
                subtitle="Type /help for commands"
            )
            console.print(welcome_panel)

            # Quick tips
            tips_text = Text()
            tips_text.append("💡 Quick Start:\n", style="bold yellow")
            tips_text.append("• Ask any question naturally - I'll route it to the best AI assistant\n")
            tips_text.append("• Use ", style="white")
            tips_text.append("/help", style="cyan")
            tips_text.append(" to see all available commands\n", style="white")
            tips_text.append("• Use ", style="white")
            tips_text.append("/summary", style="cyan")
            tips_text.append(" to get a conversation overview\n", style="white")
            tips_text.append("• Use ", style="white")
            tips_text.append("/clear", style="cyan")
            tips_text.append(" to start a fresh conversation", style="white")

            console.print(Padding(tips_text, (1, 2)))
        else:
            print("\n" + "=" * 60)
            print("🤖 RECALLR - AI-Powered Learning Assistant 🤖")
            print("=" * 60)
            print("💡 Quick Start:")
            print("• Ask any question naturally")
            print("• Type /help for commands")
            print("• Type /summary for conversation overview")
            print("• Type /clear to start fresh")
            print("=" * 60)

    def display_prompt(self) -> str:
        """Display enhanced input prompt"""
        if RICH_AVAILABLE:
            prompt_text = Text()
            prompt_text.append("🤖 ", style="bold blue")
            prompt_text.append("Recallr", style="bold cyan")
            prompt_text.append(" » ", style="bold white")
            return prompt_text
        else:
            return "🤖 Recallr » "

    def parse_command(self, user_input: str) -> tuple[Optional[str], List[str]]:
        """Parse slash commands from user input"""
        if not user_input.startswith('/'):
            return None, []

        # Remove leading slash and split into command and args
        parts = user_input[1:].split()
        if not parts:
            return None, []

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        return command, args

    async def handle_command(self, command: str, args: List[str]) -> bool:
        """Handle slash command execution"""
        if command in self.commands:
            try:
                return await self.commands[command].execute(args, self.context)
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"❌ [red]Error executing command '{command}': {e}[/red]")
                else:
                    print(f"❌ Error executing command '{command}': {e}")
                return True
        else:
            if RICH_AVAILABLE:
                console.print(f"❌ [red]Unknown command: '{command}'. Type [cyan]/help[/cyan] for available commands.[/red]")
            else:
                print(f"❌ Unknown command: '{command}'. Type /help for available commands.")
            return True

    def display_query_feedback(self, query: str, routing_info: Dict[str, Any] = None):
        """Display query processing feedback"""
        if RICH_AVAILABLE:
            # Show query being processed
            query_panel = Panel(
                f"📝 [bold white]{query}[/bold white]",
                title="Processing Query",
                border_style="blue"
            )
            console.print(query_panel)

            if routing_info:
                routing_text = Text()
                routing_text.append("🎯 Route: ", style="bold")
                routing_text.append(routing_info.get('routing', 'Unknown'), style="cyan")
                routing_text.append(f" (confidence: {routing_info.get('confidence', 0):.2f})", style="dim")
                if routing_info.get('reasoning'):
                    routing_text.append(f"\n💡 Reasoning: {routing_info['reasoning']}", style="dim")

                console.print(routing_text)
        else:
            print(f"\n📝 Processing: {query}")
            if routing_info:
                print(f"🎯 Route: {routing_info.get('routing', 'Unknown')} (confidence: {routing_info.get('confidence', 0):.2f})")
                if routing_info.get('reasoning'):
                    print(f"💡 Reasoning: {routing_info['reasoning']}")

    def display_response(self, response: str, source: str = "Assistant"):
        """Display AI response with enhanced formatting"""
        if RICH_AVAILABLE:
            # Create response panel
            response_panel = Panel(
                Markdown(response),
                title=f"🤖 {source}",
                border_style="green",
                expand=False
            )
            console.print(response_panel)
        else:
            print(f"\n{'='*60}")
            print(f"🤖 {source}:")
            print('='*60)
            print(response)
            print('='*60)

    def display_error(self, error: str):
        """Display error message with enhanced formatting"""
        if RICH_AVAILABLE:
            console.print(f"❌ [bold red]Error:[/bold red] [red]{error}[/red]")
        else:
            print(f"❌ Error: {error}")

    def display_info(self, message: str, style: str = "info"):
        """Display informational message"""
        if RICH_AVAILABLE:
            style_map = {
                "info": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red"
            }
            console.print(f"ℹ️ [{style_map.get(style, 'blue')}]{message}[/{style_map.get(style, 'blue')}]")
        else:
            icon_map = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌"
            }
            print(f"{icon_map.get(style, 'ℹ️')} {message}")

    async def get_user_input(self) -> str:
        """Get user input with enhanced prompt"""
        if RICH_AVAILABLE:
            try:
                return Prompt.ask(self.display_prompt()).strip()
            except (KeyboardInterrupt, EOFError):
                return "/exit"
        else:
            try:
                return input(f"\n{self.display_prompt()}").strip()
            except (KeyboardInterrupt, EOFError):
                return "/exit"

    def show_loading(self, message: str = "Processing..."):
        """Show loading spinner (if rich is available)"""
        if RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            )
        else:
            print(f"🔄 {message}")
            return None

# Global instance
enhanced_cli = EnhancedCLI()

def get_cli() -> EnhancedCLI:
    """Get the global CLI instance"""
    return enhanced_cli

# Export the main classes
__all__ = ['EnhancedCLI', 'SlashCommand', 'get_cli', 'console']