"""
Recallr - AI-Powered Learning Assistant

A comprehensive AI assistant for learning that provides:
- Document processing and search
- Mathematical problem solving
- Code help and programming assistance
- YouTube and web search integration
- Interactive chat interface

Author: Recallr Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Recallr Team"
__description__ = "AI-Powered Learning Assistant"

# Import main modules for easier access
try:
    from . import main
    from . import hybrid
    from . import code_search
    from . import math_search
    from . import doc_processing
    from . import youtube
except ImportError:
    # Handle relative imports when run as script
    pass

__all__ = [
    'main',
    'hybrid', 
    'code_search',
    'math_search',
    'doc_processing',
    'youtube'
]