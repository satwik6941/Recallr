#!/usr/bin/env python3
"""
Recallr CLI - Global command-line interface for the Recallr AI assistant
"""

import os
import sys
import subprocess
import importlib.metadata
import importlib.util
from pathlib import Path
import asyncio
import threading
import time
import warnings
import io
import contextlib
import logging

# Suppress specific warnings and logs for cleaner startup
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress LlamaIndex logging
logging.getLogger().setLevel(logging.CRITICAL)  # Root logger to CRITICAL
logging.getLogger('llama_index').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
logging.getLogger('llama_index.core.storage').setLevel(logging.CRITICAL)
logging.getLogger('llama_index.storage').setLevel(logging.CRITICAL)
logging.getLogger('llama_index.core.storage.kvstore').setLevel(logging.CRITICAL)
logging.getLogger('llama_index.core.storage.kvstore.simple_kvstore').setLevel(logging.CRITICAL)
logging.getLogger('google').setLevel(logging.CRITICAL)
logging.getLogger('google.generativeai').setLevel(logging.CRITICAL)

class AnimatedLoader:
    def __init__(self, message, spinner_chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        self.message = message
        self.spinner_chars = spinner_chars
        self.running = False
        self.thread = None
        self.result = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()
    
    def stop(self, result="✅", end_message=None):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line and show result
        print(f"\r{' ' * 80}\r{result} {end_message or self.message}", flush=True)
    
    def _animate(self):
        i = 0
        while self.running:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            print(f"\r{char} {self.message}...", end="", flush=True)
            time.sleep(0.1)
            i += 1

def animated_progress_bar(message, duration=2.0, width=30):
    """Show an animated progress bar"""
    print(f"\n{message}")
    for i in range(width + 1):
        percent = int((i / width) * 100)
        filled = "█" * i
        empty = "░" * (width - i)
        print(f"\r[{filled}{empty}] {percent}%", end="", flush=True)
        time.sleep(duration / width)
    print("  ✅")

def typewriter_effect(text, delay=0.03):
    """Print text with typewriter effect"""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

def display_ascii_art():
    """Display the Recallr ASCII art logo with animation"""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    ascii_lines = [
        "██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗     ██████╗",
        "██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║     ██╔══██╗",
        "██████╔╝█████╗  ██║     ███████║██║     ██║     ██████╔╝",
        "██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║     ██╔══██╗",
        "██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗██║  ██║",
        "╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝",
        "",
        "    🤖 Your AI-Powered Learning Assistant 🤖"
    ]
    
    for line in ascii_lines:
        print(line)

    print()

def get_recallr_path():
    """Get the path where Recallr source files are located"""
    # First, try to find where cli module is installed
    try:
        import cli
        cli_path = Path(cli.__file__).parent
        if (cli_path / "main.py").exists():
            return cli_path
    except ImportError:
        pass
    
    # Try to find via importlib
    try:
        spec = importlib.util.find_spec('cli')
        if spec and spec.origin:
            cli_path = Path(spec.origin).parent
            if (cli_path / "main.py").exists():
                return cli_path
    except:
        pass
    
    # Fallback: look for main.py in common locations
    possible_paths = [
        Path(__file__).parent,  # Same directory as CLI script
        Path.cwd(),             # Current working directory
    ]
    
    for path in possible_paths:
        if (path / "main.py").exists():
            return path
    
    # If not found, use the directory where this CLI script is located
    return Path(__file__).parent

def _dep_stamp_path() -> Path:
    return get_recallr_path() / ".deps_ok"

def check_dependencies():
    """Check dependencies. Uses a stamp file so the full scan only runs once after install."""
    recallr_path = get_recallr_path()
    requirements_file = recallr_path / "requirements.txt"
    stamp = _dep_stamp_path()

    if not requirements_file.exists():
        print("❌ Requirements file not found")
        return False

    # Re-check requirements mtime against the stamp — only scan when something changed
    req_mtime = requirements_file.stat().st_mtime
    if stamp.exists():
        try:
            cached_mtime = float(stamp.read_text().strip())
            if cached_mtime >= req_mtime:
                print("✅ Dependencies OK (cached)")
                return True
        except (ValueError, OSError):
            pass

    # Full scan
    loader = AnimatedLoader("Verifying dependencies")
    loader.start()

    requirements = []
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pkg = line.split('>=')[0].split('==')[0].split('[')[0].strip()
            if pkg:
                requirements.append(pkg)

    missing = []
    for pkg in requirements:
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)

    if missing:
        loader.stop("❌", f"Missing {len(missing)} dependencies")
        print(f"   Missing: {', '.join(missing)}")
        return False

    loader.stop("✅", f"All {len(requirements)} dependencies verified")
    # Write stamp so next launch skips the scan
    try:
        stamp.write_text(str(req_mtime))
    except OSError:
        pass
    return True


def invalidate_dep_cache():
    """Remove the dependency stamp so the next launch re-scans."""
    stamp = _dep_stamp_path()
    if stamp.exists():
        stamp.unlink()

def install_dependencies():
    """Install missing dependencies with animated progress"""
    loader = AnimatedLoader("Installing dependencies", "🔄⚡🔄⚡")
    loader.start()
    
    recallr_path = get_recallr_path()
    requirements_file = recallr_path / "requirements.txt"
    
    try:
        # Run pip install in background
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            loader.stop("🎉", "All dependencies installed successfully!")
            invalidate_dep_cache()
            return True
        else:
            loader.stop("❌", "Failed to install dependencies")
            return False
    except Exception as e:
        loader.stop("❌", f"Installation error: {str(e)[:50]}...")
        return False

def setup_env_file():
    """Create or update .env file with user input"""
    recallr_path = get_recallr_path()
    env_file = recallr_path / ".env"

    print("\n🔑 Setting up environment variables...")
    print("Please enter your API keys (required keys must be filled):\n")

    openai_key = input("🤖 OPENAI_API_KEY (required): ").strip()
    tavily_key = input("🔍 TAVILY_API_KEY (required): ").strip()
    groq_key = input("⚡ GROQ_API_KEY (required): ").strip()
    youtube_key = input("📺 YOUTUBE_API_KEY (optional, for YouTube search): ").strip()
    mistral_key = input("🧠 MISTRAL_API_KEY (optional, for enhanced code/math): ").strip()

    if not openai_key or not tavily_key or not groq_key:
        print("❌ OPENAI_API_KEY, TAVILY_API_KEY, and GROQ_API_KEY are all required.")
        return False

    env_content = "# Recallr Environment Variables\n"
    env_content += f"OPENAI_API_KEY={openai_key}\n"
    env_content += f"TAVILY_API_KEY={tavily_key}\n"
    env_content += f"GROQ_API_KEY={groq_key}\n"
    env_content += f"YOUTUBE_API_KEY={youtube_key}\n" if youtube_key else "# YOUTUBE_API_KEY=your_youtube_api_key_here\n"
    env_content += f"MISTRAL_API_KEY={mistral_key}\n" if mistral_key else "# MISTRAL_API_KEY=your_mistral_api_key_here\n"

    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Environment file created at: {env_file}")

        for key, val in [("OPENAI_API_KEY", openai_key), ("TAVILY_API_KEY", tavily_key),
                         ("GROQ_API_KEY", groq_key)]:
            os.environ[key] = val
        if youtube_key:
            os.environ['YOUTUBE_API_KEY'] = youtube_key
        if mistral_key:
            os.environ['MISTRAL_API_KEY'] = mistral_key

        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def load_env_file():
    """Load environment variables from .env file"""
    recallr_path = get_recallr_path()
    env_file = recallr_path / ".env"
    
    loaded_vars = {}
    
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle lines with = sign
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        if key and value:
                            loaded_vars[key] = value
                            # Set in environment if not already set
                            if key not in os.environ:
                                os.environ[key] = value
                            
            return loaded_vars
        except Exception as e:
            print(f"⚠️  Error reading .env file: {e}")
            return {}
    
    return {}

def check_environment():
    """Check if required environment variables are set with animation"""
    loader = AnimatedLoader("Validating API configuration")
    loader.start()

    recallr_path = get_recallr_path()
    env_file = recallr_path / ".env"
    
    # Load .env file first
    loaded_vars = load_env_file()
    
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "GROQ_API_KEY"]
    optional_vars = ["YOUTUBE_API_KEY", "MISTRAL_API_KEY", "MISTRAL_API_KEY_1"]
    
    missing_required = []
    found_required = []
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if value and value.strip() and value != "":
            found_required.append(var)
        else:
            missing_required.append(var)
    
    # Check optional variables
    optional_found = sum(1 for var in optional_vars if os.getenv(var) and os.getenv(var).strip())
    
    if missing_required:
        loader.stop("❌", "Missing required API keys")
        print(f"\n🔑 Missing: {', '.join(missing_required)}")
        print("\n🔧 Would you like to set them up now? (y/n): ", end="")
        
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                return setup_env_file()
            else:
                print("\n📝 To set up manually:")
                print(f"1. Create/edit .env file at: {env_file}")
                print("2. Add required keys: OPENAI_API_KEY=your_key_here")
                print("3. Add required keys: TAVILY_API_KEY=your_key_here")
                print("4. Add required keys: GROQ_API_KEY=your_key_here")
                print("5. Optionally add: YOUTUBE_API_KEY, MISTRAL_API_KEY")
                return False
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled.")
            return False
    
    # Success message with features count
    features_msg = f"Core features ready"
    if optional_found > 0:
        features_msg += f" + {optional_found} enhanced feature{'s' if optional_found > 1 else ''}"
    
    loader.stop("🔑", features_msg)
    return True

def initialize_application():
    """Initialize the Recallr application with smooth animations"""
    recallr_path = get_recallr_path()

    # Change to the Recallr directory to ensure all file operations work correctly
    original_cwd = os.getcwd()
    os.chdir(recallr_path)

    try:
        # Add the Recallr path to Python path
        sys.path.insert(0, str(recallr_path))

        # Set environment variable to indicate CLI mode
        os.environ['RECALLR_SOURCE_PATH'] = str(recallr_path)

        # Import main module (suppress noisy library output)
        loader_main = AnimatedLoader("Loading core modules")
        loader_main.start()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import main
        loader_main.stop("📦", "Core modules loaded")

        # Set quiet mode for cleaner output
        os.environ['RECALLR_QUIET_MODE'] = '1'
        os.environ['GOOGLE_API_USE_CLIENT_CERTIFICATE'] = 'false'

        # Run the main application
        asyncio.run(main.main())
        
    except ImportError as e:
        print(f"❌ Failed to import main module: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to start Recallr: {e}")
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    return True

def install_globally():
    """Install Recallr globally so it can be run from anywhere"""
    recallr_path = get_recallr_path()
    
    print("\n🌍 Global Installation Wizard")
    print("=" * 50)
    print("\nThis will install Recallr as a global command.")
    print("After installation, you can run 'recallr' from anywhere in your terminal!\n")
    
    # Step 1: Verify we're in the right directory
    setup_file = recallr_path / "setup.py"
    if not setup_file.exists():
        print("❌ setup.py not found. Please run this from the Recallr directory.")
        return False
    
    # Step 2: Check dependencies first
    print("📦 Step 1: Checking dependencies...")
    if not check_dependencies():
        print("\n📥 Installing dependencies first...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return False
    
    # Step 3: Check environment variables
    print("\n🔑 Step 2: Checking API keys...")
    if not check_environment():
        print("❌ API keys not configured")
        return False
    
    # Step 4: Install package globally
    print("\n🚀 Step 3: Installing Recallr globally...")
    loader = AnimatedLoader("Installing Recallr package globally")
    loader.start()
    
    try:
        # Install in editable mode so changes are reflected immediately
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", str(recallr_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            loader.stop("🎉", "Recallr installed globally!")
            
            print("\n" + "=" * 50)
            print("✅ Installation Complete!")
            print("=" * 50)
            print("\n📍 You can now run Recallr from anywhere:")
            print("   Just type: recallr")
            print("\n🔧 Useful commands:")
            print("   recallr          - Start the assistant")
            print("   recallr --help   - Show help")
            print("   recallr --status - Check system status")
            print("\n💡 Tip: Close and reopen your terminal for changes to take effect.")
            print("=" * 50)
            return True
        else:
            loader.stop("❌", "Installation failed")
            print(f"\n❌ Error: {result.stderr}")
            return False
            
    except Exception as e:
        loader.stop("❌", f"Installation error: {str(e)[:50]}...")
        return False

def uninstall_globally():
    """Uninstall Recallr from global installation"""
    print("\n🗑️ Uninstalling Recallr globally...")
    loader = AnimatedLoader("Removing global installation")
    loader.start()
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "recallr", "-y"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            loader.stop("✅", "Recallr uninstalled successfully")
            print("\n✅ Recallr has been removed from global installation.")
            print("💡 You can still run it locally with: python recallr_main.py")
            return True
        else:
            loader.stop("⚠️", "Uninstall completed with warnings")
            return True
            
    except Exception as e:
        loader.stop("❌", f"Uninstall error")
        print(f"Error: {e}")
        return False

def show_help():
    """Display help information"""
    print("""
🤖 Recallr CLI - AI-Powered Learning Assistant

USAGE:
    python recallr_main.py [OPTIONS]    # Local run
    recallr [OPTIONS]                   # After global installation

OPTIONS:
    --help, -h        Show this help message
    --version, -v     Show version information
    --status          Check system status (dependencies, environment)
    --install         Install Recallr globally (run 'recallr' from anywhere)
    --uninstall       Remove global installation
    --check-deps      Force a full dependency re-scan (clears cached result)

DESCRIPTION:
    Recallr is an AI-powered learning assistant that helps with:
    • Document processing and search (place PDFs in the data/ folder)
    • Mathematical problem solving
    • Code help and programming assistance
    • YouTube and web search integration
    • Interactive chat interface with slash commands

INTERACTIVE COMMANDS:
    /mode [name]   - Switch pipeline (AUTO, ACADEMIC_RAG, MATH, CODE, GENERAL)
    /summary, /s   - Generate conversation summary
    /clear, /c     - Clear conversation history
    /help, /h      - Show available commands
    /status        - Show system status
    /exit, /quit   - Exit the application

SETUP:
    Required keys: OPENAI_API_KEY, TAVILY_API_KEY, GROQ_API_KEY
    Optional keys: YOUTUBE_API_KEY, MISTRAL_API_KEY

    1. Run 'recallr' — it will prompt you to enter keys and create .env
    2. Place your PDF documents in the data/ folder
    3. Ask questions — AUTO mode routes to the best pipeline automatically

For more information, visit: https://github.com/satwik6941/Recallr
""")

def show_version():
    """Display version information"""
    print(f"Recallr CLI v1.0.0")
    print(f"Python {sys.version}")
    print(f"Working directory: {get_recallr_path()}")

def check_system_files():
    """Check critical system files and directories (silent)"""
    recallr_path = get_recallr_path()
    critical_files = {
        'main.py': 'Main application script',
        'hybrid.py': 'Hybrid search module',
        'code_search.py': 'Code search functionality',
        'math_search.py': 'Math search functionality', 
        'doc_processing.py': 'Document processing module',
        'youtube.py': 'YouTube integration',
        'requirements.txt': 'Dependencies list'
    }
    
    critical_dirs = {
        'data': 'PDF documents storage',
        'storage': 'Vector store and indexes'
    }
    
    files_ok = True
    dirs_ok = True
    
    for file_name, description in critical_files.items():
        file_path = recallr_path / file_name
        if not file_path.exists():
            files_ok = False
    
    for dir_name, description in critical_dirs.items():
        dir_path = recallr_path / dir_name
        if not dir_path.exists():
            dirs_ok = False
    
    return files_ok and dirs_ok

def check_python_environment():
    """Check Python environment and version (silent)"""
    python_version = sys.version_info
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        return False
    
    # Check pip
    try:
        import pip
        return True
    except ImportError:
        return False

def check_disk_space():
    """Check available disk space (silent)"""
    recallr_path = get_recallr_path()
    try:
        import shutil
        total, used, free = shutil.disk_usage(recallr_path)
        free_gb = free // (1024**3)
        return free_gb >= 1  # At least 1 GB free
    except Exception:
        return True  # Assume OK if we can't check

def check_status():
    """Check and display comprehensive system status"""
    display_ascii_art()
    print("🔍 Comprehensive System Status Check")
    print("=" * 40)
    print("💡 Enhanced CLI with slash commands available!")
    print("   Use /help inside the application for interactive commands")
    
    # Check Python environment
    python_ok = check_python_environment()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check system files
    files_ok = check_system_files()
    
    # Check disk space
    disk_ok = check_disk_space()
    
    # Check environment variables
    env_ok = check_environment()
    
    # Summary
    recallr_path = get_recallr_path()
    print(f"\n📍 Installation Summary:")
    print(f"   📁 Source directory: {recallr_path}")
    print(f"   🐍 Python environment: {'✅' if python_ok else '❌'}")
    print(f"   📦 Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   📄 System files: {'✅' if files_ok else '❌'}")
    print(f"   💾 Disk space: {'✅' if disk_ok else '❌'}")
    print(f"   🔑 Environment variables: {'✅' if env_ok else '❌'}")
    
    print("\n📋 Overall Status:")
    if all([python_ok, deps_ok, files_ok, disk_ok, env_ok]):
        print("🎉 All systems ready! Recallr is fully operational!")
        print("✨ Enhanced CLI with slash commands is ready")
        print("   Available commands: /summary, /clear, /help, /status, /exit")
    else:
        issues = []
        if not python_ok: issues.append("Python environment")
        if not deps_ok: issues.append("dependencies")
        if not files_ok: issues.append("system files")
        if not disk_ok: issues.append("disk space")
        if not env_ok: issues.append("environment variables")
        print(f"❌ Issues found with: {', '.join(issues)}")
        print("Run 'recallr' to start the setup process.")

def main():
    """Main CLI entry point"""
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h']:
            show_help()
            return
        elif arg in ['--version', '-v']:
            show_version()
            return
        elif arg == '--status':
            check_status()
            return
        elif arg == '--install':
            success = install_globally()
            sys.exit(0 if success else 1)
        elif arg == '--uninstall':
            success = uninstall_globally()
            sys.exit(0 if success else 1)
        elif arg == '--check-deps':
            invalidate_dep_cache()
            success = check_dependencies()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    try:
        print("🚀 Welcome to Recallr!")

        # Step 1: Python environment check
        loader1 = AnimatedLoader("Verifying system requirements")
        loader1.start()
        python_ok = check_python_environment()
        if not python_ok:
            loader1.stop("❌", "Python environment incompatible")
            sys.exit(1)
        loader1.stop("✅", "System requirements met")

        # Step 2: File system check
        loader2 = AnimatedLoader("Scanning installation files")
        loader2.start()
        files_ok = check_system_files()
        if not files_ok:
            loader2.stop("❌", "Critical files missing")
            sys.exit(1)
        loader2.stop("✅", "Installation verified")

        # Step 3: Dependencies
        if not check_dependencies():
            if not install_dependencies():
                print("\n❌ Dependency installation failed")
                sys.exit(1)

        # Step 4: Disk space
        loader4 = AnimatedLoader("Checking available storage")
        loader4.start()
        disk_ok = check_disk_space()
        loader4.stop("✅" if disk_ok else "⚠️", "Storage validated")

        # Step 5: Environment variables / API keys
        if not check_environment():
            print("\n❌ Environment setup cancelled")
            sys.exit(1)

        # Show ASCII art and launch
        display_ascii_art()
        initialize_application()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using Recallr!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()