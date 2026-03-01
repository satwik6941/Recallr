# 🤖 Recallr - AI-Powered Learning Assistant

<div align="center">

```
██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗     ██████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║     ██╔══██╗
██████╔╝█████╗  ██║     ███████║██║     ██║     ██████╔╝
██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║     ██╔══██╗
██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
```

**Your AI-Powered Learning Assistant**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/satwik6941/Recallr)

</div>

## 📖 Introduction

Recallr is a comprehensive AI-powered learning assistant designed to enhance your learning experience through intelligent document processing, mathematical problem solving, code assistance, and multi-modal search capabilities. Built with state-of-the-art AI models and retrieval-augmented generation (RAG) technology, Recallr transforms how you interact with academic content.

Whether you're a student working through complex coursework, a researcher analyzing documents, or a developer seeking coding assistance, Recallr provides intelligent, context-aware responses tailored to your learning needs.

## ✨ Features

### 🔍 **Intelligent Document Processing**

- **PDF Analysis**: Automatically processes and indexes PDF documents from your `data/` folder
- **AI-Optimized Chunking**: Uses machine learning to determine optimal document segmentation
- **Smart Caching**: Efficiently caches processed documents to avoid redundant processing
- **Academic Focus**: Specialized for academic and technical documents

### 🧠 **Advanced AI Capabilities**

- **Hybrid Retrieval**: Combines vector search, keyword search, and BM25 ranking for optimal results
- **Context-Aware Responses**: Maintains conversation history for coherent multi-turn interactions
- **Auto Query Routing**: Automatically routes queries to specialized assistants (Math, Code, Academic, or General)
- **Multi-Model Integration**: Leverages OpenAI, Mistral, and Groq models for diverse AI perspectives

### 📊 **Specialized Search Modules**

- **📚 Academic RAG**: Document-based question answering with academic focus
- **🔢 Mathematics Assistant**: Specialized mathematical problem solving and explanations
- **💻 Code Assistant**: Programming help, code review, and technical guidance
- **🌐 General Web Search**: Real-time web search powered by Tavily + OpenAI for current information
- **📺 YouTube Search**: Educational video discovery and content analysis

### 🎨 **User Experience**

- **Enhanced CLI**: Beautiful Rich-powered interface with syntax highlighting and panels
- **Slash Commands**: Interactive commands (`/summary`, `/clear`, `/help`, `/status`, `/exit`)
- **Animated Startup**: Clean, professional startup experience with progress indicators
- **Quiet Mode**: Suppresses verbose technical output for smooth operation
- **Conversation History**: Automatic saving and loading of chat sessions

## 🚀 Installation Guide

### Prerequisites

- **Python 3.8 or higher**
- **pip package manager**
- **Internet connection** (for AI model access)

### Option 1: Quick Local Run (Recommended for First-Time Setup)

1. **Clone the Repository**

   ```bash
   git clone https://github.com/satwik6941/Recallr.git
   cd Recallr
   ```

2. **Set up environment variables**

   Create a `.env` file in the project root (see [Environment Variables](#-environment-variables) section):

   ```bash
   cp .env.local .env
   # Then edit .env and fill in your actual API keys
   ```

3. **Run Recallr** (Auto-setup)

   ```bash
   python recallr_main.py
   ```

   The first run will automatically:
   - ✅ Check system requirements
   - 📦 Install required dependencies
   - 🔑 Validate environment variables
   - 📁 Create necessary directories
   - 🚀 Launch the assistant

### Option 2: Global Installation (Run from Anywhere)

For convenient access to Recallr from any directory in your terminal:

1. **Clone and Navigate**

   ```bash
   git clone https://github.com/satwik6941/Recallr.git
   cd Recallr
   ```

2. **Install Globally**

   ```bash
   python recallr_main.py --install
   ```

   This will:
   - ✅ Verify system requirements
   - 📦 Install all dependencies
   - 🔑 Configure API keys
   - 🌍 Install Recallr as a global command

3. **Use from Anywhere**

   After installation, open a new terminal and run:

   ```bash
   recallr
   ```

   **Useful Global Commands:**
   - `recallr` — Start the assistant
   - `recallr --help` — Show help information
   - `recallr --status` — Check system status
   - `recallr --version` — Show version
   - `recallr --uninstall` — Remove global installation

### Manual Installation (Advanced Users)

```bash
# Clone repository
git clone https://github.com/satwik6941/Recallr.git
cd Recallr

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys (see below)
cp .env.local .env

# Install globally (optional)
pip install -e .

# Run
python recallr_main.py
```

## 🔐 Environment Variables

Create a `.env` file in the project root. Use `.env.local` as a template:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Required | Powers the core LLM responses |
| `TAVILY_API_KEY` | ✅ Required | Powers the general web search module |
| `GROQ_API_KEY` | Optional | High-speed inference for math/code modules |
| `MISTRAL_API_KEY` | Optional | Alternative LLM for math/code modules |
| `YOUTUBE_API_KEY` | Optional | YouTube search and content discovery |
| `GOOGLE_API_KEY` | Optional | Google services integration |
| `GOOGLE_CSE_ID` | Optional | Google Custom Search Engine |

## 📁 Project Structure

```bash
Recallr/
├── 📄 README.md              # This file
├── 🚀 recallr_main.py        # Main CLI entry point & global installer
├── 🧠 main.py                # Core application logic & query routing
├── 🖥️  cli_interface.py       # Enhanced Rich-powered CLI with slash commands
├── 🔍 hybrid.py              # Hybrid retrieval system (vector + BM25)
├── 💻 code_search.py         # Code assistance module
├── 🔢 math_search.py         # Mathematics assistant
├── 🌐 general_search.py      # General web search (Tavily + OpenAI)
├── 📚 doc_processing.py      # Document processing & RAG pipeline
├── 📺 youtube.py             # YouTube integration
├── ⚙️  requirements.txt       # Python dependencies
├── 🔧 setup.py               # Package setup for global install
├── 🔒 .env.local             # Environment variable template
├── 📁 data/                  # Your PDF documents (create this)
├── 💾 storage/               # AI indexes and caches
└── 🌍 .env                   # Environment variables (create from .env.local)
```

## 💬 Using the Assistant

1. **Add Documents**: Place PDF files in the `data/` folder for academic RAG
2. **Start Recallr**: Run `recallr` or `python recallr_main.py`
3. **Ask Questions**: Type your questions naturally — Recallr auto-routes to the best module
4. **Slash Commands** (inside the app):
   - `/help` or `/h` — Show available commands
   - `/summary` or `/s` — Generate conversation summary
   - `/clear` or `/c` — Clear conversation history
   - `/status` — Show system status
   - `/exit` or `/quit` — Exit the application

## 🛠️ Tech Stack

### Core AI & Search

- **[OpenAI](https://openai.com/)** — Primary LLM for reasoning and generation (`gpt-4o-mini`)
- **[Tavily](https://tavily.com/)** — AI-powered real-time web search
- **[Groq](https://groq.com/)** — High-speed inference for math/code modules
- **[Mistral AI](https://mistral.ai/)** — Efficient language models
- **[LlamaIndex](https://github.com/run-llama/llama_index)** — RAG framework for document processing
- **[Sentence Transformers](https://www.sbert.net/)** — Semantic text embeddings
- **[BM25](https://github.com/dorianbrown/rank_bm25)** — Keyword ranking for hybrid search

### Document Processing

- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)** — PDF processing and text extraction
- **[PyPDF2](https://pypdf2.readthedocs.io/)** — Additional PDF manipulation
- **[docx2txt](https://github.com/ankushshah89/python-docx2txt)** — Word document extraction
- **[Transformers](https://huggingface.co/transformers/)** — NLP models from Hugging Face
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[NLTK](https://www.nltk.org/)** — Natural Language Processing toolkit

### CLI & UX

- **[Rich](https://rich.readthedocs.io/)** — Beautiful CLI formatting, panels, and syntax highlighting
- **[Typer](https://typer.tiangolo.com/)** — Modern CLI framework
- **[Click](https://click.palletsprojects.com/)** — Command line interface toolkit
- **[python-dotenv](https://python-dotenv.readthedocs.io/)** — Environment variable management

### Integrations

- **[YouTube Data API](https://developers.google.com/youtube/v3)** — YouTube search and content access
- **[Google API Client](https://github.com/googleapis/google-api-python-client)** — Google services integration
- **[Requests](https://requests.readthedocs.io/)** — HTTP library

---

<div align="center">

⭐ **Star this repo if you find it helpful!** ⭐

[🐛 Report Bug](https://github.com/satwik6941/Recallr/issues) · [✨ Request Feature](https://github.com/satwik6941/Recallr/issues) · [🤝 Contribute](https://github.com/satwik6941/Recallr/pulls)

</div>