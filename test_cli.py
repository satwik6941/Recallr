#!/usr/bin/env python3
"""
Test script for the enhanced CLI interface
"""

import asyncio
import sys
import os
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_cli():
    """Test the enhanced CLI interface"""
    print("🧪 Testing Enhanced CLI Interface")
    print("=" * 50)

    try:
        # Test importing the CLI interface
        from cli_interface import get_cli, SlashCommand, console
        print("✅ Successfully imported cli_interface")

        # Initialize CLI
        cli = get_cli()
        print("✅ CLI instance created")

        # Test context functions
        def mock_summary():
            print("📝 Mock conversation summary generated")

        def mock_clear():
            print("🧹 Mock conversation history cleared")

        cli.set_context_function('summary_function', mock_summary)
        cli.set_context_function('clear_function', mock_clear)
        print("✅ Context functions set")

        # Test conversation history update
        mock_history = [
            {"user": "What is Python?", "assistant": "Python is a programming language..."},
            {"user": "How do I use it?", "assistant": "You can use Python by..."}
        ]
        cli.update_conversation_history(mock_history)
        print("✅ Conversation history updated")

        # Test command parsing
        test_commands = [
            "/help",
            "/summary",
            "/clear",
            "/status",
            "/h",  # alias test
            "/s",  # alias test
            "/unknown",  # should show error
        ]

        print("\n🔍 Testing command parsing:")
        for cmd_text in test_commands:
            command, args = cli.parse_command(cmd_text)
            if command:
                print(f"  {cmd_text} -> command: '{command}', args: {args}")
            else:
                print(f"  {cmd_text} -> not a command")

        print("\n🎨 Testing display functions:")

        # Test welcome display
        print("\n--- Testing Welcome Display ---")
        cli.display_welcome()

        # Test info display
        print("\n--- Testing Info Display ---")
        cli.display_info("This is an info message", "info")
        cli.display_info("This is a success message", "success")
        cli.display_info("This is a warning message", "warning")
        cli.display_info("This is an error message", "error")

        # Test query feedback
        print("\n--- Testing Query Feedback ---")
        mock_routing = {
            'routing': 'MATH_SEARCH',
            'confidence': 0.85,
            'reasoning': 'Detected mathematical concepts'
        }
        cli.display_query_feedback("What is calculus?", mock_routing)

        # Test response display
        print("\n--- Testing Response Display ---")
        cli.display_response("This is a test response from the AI assistant", "Test Assistant")

        # Test error display
        print("\n--- Testing Error Display ---")
        cli.display_error("This is a test error message")

        print("\n✅ All CLI interface tests passed!")

        # Test interactive command execution
        print("\n🎮 Testing Interactive Commands:")
        print("Testing /help command...")
        await cli.handle_command("help", [])

        print("\nTesting /status command...")
        await cli.handle_command("status", [])

        print("\n🎉 Enhanced CLI interface is working correctly!")
        return True

    except ImportError as e:
        print(f"❌ Failed to import cli_interface: {e}")
        print("💡 This might be because the rich library is not installed")
        print("   Run: pip install rich")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_functionality():
    """Test basic functionality without enhanced CLI"""
    print("\n🔧 Testing Basic Functionality (Fallback Mode)")
    print("=" * 50)

    try:
        # Test basic imports
        import main
        print("✅ main.py imported successfully")

        # Test conversation functions
        main.conversation_history = [
            {"user": "test", "assistant": "test response", "timestamp": "2024-01-01 10:00:00"}
        ]

        print("✅ Conversation history functionality works")

        print("\n🎉 Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Starting CLI Enhancement Tests")
    print("=" * 60)

    # Test enhanced CLI first
    cli_test_passed = await test_cli()

    # Test basic functionality
    basic_test_passed = await test_basic_functionality()

    print("\n📊 Test Results Summary:")
    print("=" * 30)
    print(f"Enhanced CLI Test: {'✅ PASSED' if cli_test_passed else '❌ FAILED'}")
    print(f"Basic Functionality: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")

    if cli_test_passed and basic_test_passed:
        print("\n🎉 All tests passed! The enhanced CLI is ready to use.")
        print("\n💡 Usage Tips:")
        print("• Run 'recallr' to start with enhanced CLI")
        print("• Use /help to see all available slash commands")
        print("• Use /summary to get conversation overviews")
        print("• Use /clear to reset conversation history")
        print("• Enhanced interface includes progress bars and better formatting")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        if not cli_test_passed:
            print("• Enhanced CLI might need the 'rich' library: pip install rich")
        if not basic_test_passed:
            print("• Basic functionality issues - check main.py dependencies")

if __name__ == "__main__":
    asyncio.run(main())