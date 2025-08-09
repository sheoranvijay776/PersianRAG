# RAGfa - Persian RAG System

A Retrieval Augmented Generation (RAG) system built with LangChain that supports both web scraping and local text file processing, with Persian language support.


## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

#### For Text Files (MVP Version)
```bash
python mvp_rag.py
```
This will automatically load `doc.txt` and start an interactive Q&A session.

#### For Web Content (Full Version)
```bash
python langrag.py
```
This loads content from a default web URL and provides example queries.

### 3. Custom Usage

```python
from langrag import RAGSystem

rag = RAGSystem()

rag.index_docs("your-document.txt")  # Text file
# or
rag.index_docs(["https://example.com"])  # Web URLs

rag.build_graph()

result = rag.query("Your question here")
```


## Core Components

### RAGSystem Class
- **Document Loading**: Web scraping with BeautifulSoup and text file loading
- **Text Splitting**: Recursive character text splitter with configurable chunk size
- **Vector Store**: FAISS vector database for semantic search
- **LLM Integration**: OpenAI-compatible API through Metis AI
- **Graph Pipeline**: LangGraph-based retrieve-and-generate workflow

### Supported Document Types
- **Web Pages**: Automatic content extraction from HTML
- **Text Files**: UTF-8 encoded text documents
- **Persian Content**: Full support for Persian/Farsi language

## Configuration

The system uses Metis AI API with a pre-configured API key. To use your own:

```python
import os
os.environ["METIS_API_KEY"] = "your-api-key-here"
```

## Dependencies

- LangChain & LangGraph for RAG pipeline
- OpenAI for embeddings and chat models
- FAISS for vector storage
- BeautifulSoup4 for web scraping
- Python 3.9+


