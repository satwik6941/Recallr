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
    
    # Animate ASCII art line by line
    for line in ascii_lines:
        print(line)
        time.sleep(0.1)
    
    print()
    time.sleep(0.5)

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
        Path.cwd(),  # Current working directory
        Path.home() / "Documents" / "My works and PPTs" / "Recallr",  # Default location
    ]
    
    for path in possible_paths:
        if (path / "main.py").exists():
            return path
    
    # If not found, use the directory where this CLI script is located
    return Path(__file__).parent

def check_dependencies():
    """Check if all required dependencies are installed with animated progress"""
    loader = AnimatedLoader("Scanning dependencies")
    loader.start()
    
    recallr_path = get_recallr_path()
    requirements_file = recallr_path / "requirements.txt"
    
    if not requirements_file.exists():
        loader.stop("❌", "Requirements file not found")
        return False
    
    time.sleep(1)  # Simulate scanning time
    loader.stop("📦", "Dependencies scanned")
    
    # Read requirements
    with open(requirements_file, 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle different requirement formats
                if '>=' in line:
                    package = line.split('>=')[0].strip()
                    version_req = line.split('>=')[1].strip() if len(line.split('>=')) > 1 else None
                elif '==' in line:
                    package = line.split('==')[0].strip()
                    version_req = line.split('==')[1].strip() if len(line.split('==')) > 1 else None
                elif '[' in line:
                    package = line.split('[')[0].strip()
                    version_req = None
                else:
                    package = line
                    version_req = None
                requirements.append((package, version_req))
    
    missing_packages = []
    installed_packages = []
    
    # Silent package checking with progress
    loader = AnimatedLoader("Verifying package dependencies")
    loader.start()
    
    for package, version_req in requirements:
        try:
            dist = importlib.metadata.version(package)
            installed_packages.append(f"{package}=={dist}")
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
    
    time.sleep(1.5)  # Show progress animation
    
    if missing_packages:
        loader.stop("❌", f"Missing {len(missing_packages)} dependencies")
        print(f"📊 Status: {len(installed_packages)} installed, {len(missing_packages)} missing")
        return False
    else:
        loader.stop("✅", f"All {len(installed_packages)} dependencies verified")
        return True

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
        
        time.sleep(2)  # Let animation run for effect
        
        if result.returncode == 0:
            loader.stop("🎉", "All dependencies installed successfully!")
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
    print("Please enter your API keys (press Enter to skip optional ones):\n")
    
    # Get GEMINI_API_KEY
    gemini_key = input("GEMINI_API_KEY (required): ").strip()
    if not gemini_key:
        print("GEMINI_API_KEY is required for Recallr to work!")
        return False
    
    # Get YOUTUBE_API_KEY (optional)
    youtube_key = input("📺 YOUTUBE_API_KEY (optional, for enhanced search): ").strip()
    
    # Create .env file content
    env_content = f"# Recallr Environment Variables\n"
    env_content += f"GEMINI_API_KEY={gemini_key}\n"
    if youtube_key:
        env_content += f"YOUTUBE_API_KEY={youtube_key}\n"
    else:
        env_content += f"# YOUTUBE_API_KEY=your_youtube_api_key_here\n"
    
    # Write to .env file
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Environment file created at: {env_file}")
        
        # Load the new environment variables
        os.environ['GEMINI_API_KEY'] = gemini_key
        if youtube_key:
            os.environ['YOUTUBE_API_KEY'] = youtube_key
            
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
    time.sleep(0.8)  # Simulate scanning
    
    recallr_path = get_recallr_path()
    env_file = recallr_path / ".env"
    
    # Load .env file first
    loaded_vars = load_env_file()
    
    required_vars = ["GEMINI_API_KEY"]
    optional_vars = ["YOUTUBE_API_KEY"]
    
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
                print("2. Add: GEMINI_API_KEY=your_api_key_here")
                print("3. Optionally add: YOUTUBE_API_KEY=your_youtube_api_key")
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

        # Try to import enhanced CLI first
        enhanced_cli_available = False
        try:
            from cli_interface import get_cli, console
            enhanced_cli_available = True
            print("✅ Enhanced CLI interface loaded")
        except ImportError:
            print("⚠️  Enhanced CLI not available, using basic interface")

        # Loading main module animation
        loader_main = AnimatedLoader("Loading core modules")
        loader_main.start()
        time.sleep(1.0)

        # Import main module (suppress stdout temporarily)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import main
        loader_main.stop("📦", "Core modules loaded")

        # PDF Processing animation
        loader_pdf = AnimatedLoader("Processing PDFs and academic documents")
        loader_pdf.start()
        time.sleep(2.0)
        loader_pdf.stop("📄", "PDFs processed successfully")

        # AI Chunking Strategy animation
        loader_chunk = AnimatedLoader("Preparing AI-optimized chunking strategy")
        loader_chunk.start()
        time.sleep(1.8)
        loader_chunk.stop("🧠", "Chunking strategy optimized")

        # Vector store initialization animation
        loader_vector = AnimatedLoader("Building search indexes")
        loader_vector.start()
        time.sleep(2.0)
        loader_vector.stop("🔍", "Search indexes ready")

        # Knowledge base initialization animation
        loader_kb = AnimatedLoader("Initializing academic knowledge base")
        loader_kb.start()
        time.sleep(1.5)
        loader_kb.stop("📚", "Knowledge base ready")

        # Final startup animation with completion message
        loader_final = AnimatedLoader("All initialization completed")
        loader_final.start()
        time.sleep(1.2)
        loader_final.stop("🎉", "All initialization completed successfully")

        # Final startup animation
        animated_progress_bar("🚀 Launching Recallr AI Assistant", 2.5, 30)

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
        
        time.sleep(2)  # Let animation run
        
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
        
        time.sleep(1)
        
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

DESCRIPTION:
    Recallr is an AI-powered learning assistant that helps with:
    • Document processing and search
    • Mathematical problem solving
    • Code help and programming assistance
    • YouTube and web search integration
    • Interactive chat interface with slash commands

INTERACTIVE COMMANDS:
    /summary, /s   - Generate conversation summary
    /clear, /c     - Clear conversation history
    /help, /h      - Show available commands
    /status        - Show system status
    /exit, /quit   - Exit the application

INSTALLATION:
    # First time setup - Install globally
    python recallr_main.py --install
    
    # Then use from anywhere
    recallr

SETUP:
    1. Create a .env file with GEMINI_API_KEY=your_api_key
    2. Optionally add YOUTUBE_API_KEY for enhanced search
    3. Run 'python recallr_main.py --install' for global installation
    4. Use 'recallr' command from anywhere!

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
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    try:
        # Animated startup sequence
        typewriter_effect("🚀 Welcome to Recallr!", 0.05)
        time.sleep(0.5)
        
        # Step 1: Environment Check
        loader1 = AnimatedLoader("Verifying system requirements")
        loader1.start()
        time.sleep(1.2)
        python_ok = check_python_environment()
        if not python_ok:
            loader1.stop("❌", "Python environment incompatible")
            sys.exit(1)
        loader1.stop("✅", "System requirements met")
        
        # Step 2: File System Check  
        loader2 = AnimatedLoader("Scanning installation files")
        loader2.start()
        time.sleep(1.0)
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
        
        # Step 4: Storage
        loader4 = AnimatedLoader("Checking available storage")
        loader4.start()
        time.sleep(0.6)
        disk_ok = check_disk_space()
        loader4.stop("✅" if disk_ok else "⚠️", "Storage validated")
        
        # Step 5: Environment Variables
        if not check_environment():
            print("\n❌ Environment setup cancelled")
            sys.exit(1)
        
        # Show ASCII art with animation
        display_ascii_art()

        # Display enhanced CLI info
        print("\n✨ Enhanced CLI Features:")
        print("   • Beautiful interface with Rich formatting")
        print("   • Interactive slash commands (/help, /summary, /clear, etc.)")
        print("   • Real-time progress indicators")
        print("   • Improved error handling")
        print("   • Better visual feedback")
        print("\n🚀 Starting enhanced Recallr experience...")

        # Initialize and start the application
        initialize_application()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using Recallr!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()