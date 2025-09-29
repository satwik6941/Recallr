#!/usr/bin/env python3
"""
Setup script for Recallr CLI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else "Recallr - AI-Powered Learning Assistant"

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle comments in requirements
                if '#' in line:
                    line = line.split('#')[0].strip()
                requirements.append(line)

setup(
    name="recallr",
    version="1.0.0",
    description="AI-Powered Learning Assistant with document processing, math solving, and code help",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Recallr Team",
    author_email="",
    url="https://github.com/satwik6941/Recallr",
    py_modules=[
        'recallr_main',
        'main', 
        'hybrid', 
        'code_search', 
        'math_search', 
        'doc_processing', 
        'youtube'
    ],
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'recallr=recallr_main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Students",
        "Intended Audience :: Developers",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai, education, learning, assistant, documents, math, coding",
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.json', '*.env.example'],
    },
    data_files=[
        ('', ['requirements.txt']),
    ],
    zip_safe=False,
)