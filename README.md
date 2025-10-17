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
- **Query Routing**: Automatically routes queries to specialized assistants (Math, Code, or Academic)
- **Multi-Model Integration**: Leverages both Gemini and Groq models for diverse AI perspectives

### 📊 **Specialized Search Modules**

- **📚 Academic RAG**: Document-based question answering with academic focus
- **🔢 Mathematics Assistant**: Specialized mathematical problem solving and explanations
- **💻 Code Assistant**: Programming help, code review, and technical guidance
- **🌐 Web Integration**: Real-time web search for current information
- **📺 YouTube Search**: Educational video discovery and content analysis

### 🎨 **User Experience**

- **Beautiful Animations**: Clean, professional startup experience with progress indicators
- **Quiet Mode**: Suppresses verbose technical output for smooth operation
- **Interactive CLI**: Intuitive command-line interface with helpful prompts
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

2. **Run Recallr** (Auto-setup)

   ```bash
   python recallr_main.py
   ```

   The first run will automatically:
   - ✅ Check system requirements
   - 📦 Install required dependencies  
   - 🔑 Set up environment variables
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
   - `recallr` - Start the assistant
   - `recallr --help` - Show help information
   - `recallr --status` - Check system status
   - `recallr --version` - Show version
   - `recallr --uninstall` - Remove global installation

### Manual Installation (Advanced Users)

If you prefer manual control:

```bash
# Clone repository
git clone https://github.com/satwik6941/Recallr.git
cd Recallr

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
# (See Environment Variables section below)

# Install globally (optional)
pip install -e .

# Run
python recallr_main.py
```

## 🔐 Environment Variables

Recallr requires specific API keys to function. Create a `.env` file in the project root with the following variables in the .env.local file```

## 📁 Project Structure

```bash
Recallr/
├── 📄 README.md              # This file
├── 🚀 recallr_main.py        # Main CLI entry point
├── 🧠 main.py                # Core application logic
├── 🔍 hybrid.py              # Hybrid retrieval system
├── 💻 code_search.py         # Code assistance module
├── 🔢 math_search.py         # Mathematics assistant
├── 📚 doc_processing.py      # Document processing
├── 📺 youtube.py             # YouTube integration
├── ⚙️ requirements.txt       # Python dependencies
├── 🔧 setup.py               # Package setup
├── 📁 data/                  # Your PDF documents (create this)
├── 💾 storage/               # AI indexes and caches
└── 🌍 .env                   # Environment variables (create this)
```

### Using the Assistant

1. **Add Documents**: Place PDF files in the `data/` folder
2. **Start Recallr**: Run `recallr` in your terminal
3. **Ask Questions**: Type your questions naturally
4. **Special Commands**:
   - `quit` or `q` - Exit the assistant
   - `summary` - Get conversation summary
   - `refresh` - Reload document processing

## �️ Tech Stack

Recallr is built with cutting-edge technologies and libraries to deliver powerful AI capabilities:

- **[LlamaIndex](https://github.com/run-llama/llama_index)** - Advanced RAG framework for document processing
- **[LlamaIndex Core](https://github.com/run-llama/llama_index)** - Core functionality for indexing and retrieval
- **[Transformers](https://huggingface.co/transformers/)** - State-of-the-art NLP models from Hugging Face
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic text embeddings
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[NLTK](https://www.nltk.org/)** - Natural Language Processing toolkit
- **[BM25](https://github.com/dorianbrown/rank_bm25)** - Best Matching ranking algorithm
- **[LlamaIndex BM25 Retriever](https://docs.llamaindex.ai/en/stable/)** - BM25 integration for hybrid search
- **[LlamaIndex Embeddings](https://docs.llamaindex.ai/en/stable/)** - Vector embeddings for semantic search
- **[Hugging Face Embeddings](https://huggingface.co/models)** - Pre-trained embedding models
- **[Google Gemini](https://ai.google.dev/)** - Advanced language model for reasoning and generation
- **[Groq](https://groq.com/)** - High-speed AI inference platform
- **[Mistral AI](https://mistral.ai/)** - Efficient language models
- **[Google Generative AI](https://ai.google.dev/)** - Google's generative AI platform
- **[Google API Client](https://github.com/googleapis/google-api-python-client)** - Google services integration
- **[Google Auth](https://google-auth.readthedocs.io/)** - Google authentication libraries
- **[Requests](https://requests.readthedocs.io/)** - HTTP library for web requests
- **[YouTube Data API](https://developers.google.com/youtube/v3)** - YouTube search and content access
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)** - PDF processing and text extraction
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - Additional PDF manipulation capabilities
- **[docx2txt](https://github.com/ankushshah89/python-docx2txt)** - Word document text extraction
- **[python-magic](https://github.com/ahupp/python-magic)** - File type detection
- **[Typer](https://typer.tiangolo.com/)** - Modern CLI framework
- **[Rich](https://rich.readthedocs.io/)** - Rich text and beautiful formatting
- **[Click](https://click.palletsprojects.com/)** - Command line interface creation toolkit
- **[python-dotenv](https://python-dotenv.readthedocs.io/)** - Environment variable management
- **[structlog](https://www.structlog.org/)** - Structured logging
- **[asyncio](https://docs.python.org/3/library/asyncio.html)** - Asynchronous programming support

### 🚀 **Core Technologies**

- **[LlamaIndex](https://github.com/run-llama)** - For the exceptional RAG framework that powers our document processing
- **[Google AI](https://ai.google.dev/)** - For Gemini's powerful language capabilities and Generative AI platform
- **[Groq](https://groq.com/)** - For lightning-fast AI inference and high-performance computing
- **[Hugging Face](https://huggingface.co/)** - For Transformers library and pre-trained models ecosystem
- **[Mistral AI](https://mistral.ai/)** - For efficient and powerful language models
- **[Sentence Transformers](https://www.sbert.net/)** - For semantic similarity and embedding research
- **[PyTorch](https://pytorch.org/)** - For the foundational deep learning framework
- **[Typer](https://typer.tiangolo.com/)** & **[Rich](https://rich.readthedocs.io/)** - For creating beautiful command-line interfaces
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - For excellent PDF processing capabilities

---

<div align="center">

⭐ **Star this repo if you find it helpful!** ⭐

[🐛 Report Bug](https://github.com/satwik6941/Recallr/issues) · [✨ Request Feature](https://github.com/satwik6941/Recallr/issues) · [🤝 Contribute](https://github.com/satwik6941/Recallr/pulls)

</div>
