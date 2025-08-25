# PersianRAG — Persian QA using Retrieval-Augmented Generation

[![Releases](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/sheoranvijay776/PersianRAG/releases)  
![PersianRAG banner](https://images.unsplash.com/photo-1526378724545-5f9d9bd74f5a?ixlib=rb-4.0.3&q=80&w=1200&auto=format&fit=crop&sat=-100)

A smart question-answering system for Persian documents using Retrieval Augmented Generation (RAG). Use the Releases page to download the packaged release file and run the included installer and demo scripts: https://github.com/sheoranvijay776/PersianRAG/releases

Table of contents
- Features
- Quick links
- Setup
  - Requirements
  - Install from releases
  - Local installation
  - Docker
- Core concepts
  - Retrieval
  - Augmented generation
  - Indexing and embeddings
- Usage
  - Command-line demo
  - Python API examples
  - Web UI
- Data preparation
  - Supported document types
  - Persian text tips
- Models and components
- Performance and tuning
- Testing and benchmarks
- Deployment
- Contribute
- License
- Acknowledgements
- Changelog and releases

Features
- End-to-end Persian question answering with RAG.
- Support for dense retrieval (FAISS) and sparse retrieval (BM25).
- Embedding adapters for Persian LLMs.
- Local and server modes. Use it on a laptop or on a cloud VM.
- Simple CLI and a small web UI for demos.
- Tools for indexing PDFs, DOCX, HTML, and plain text.
- Query caching, context window control, and prompt templates.

Quick links
- Releases and packaged installers: https://github.com/sheoranvijay776/PersianRAG/releases
- Issues and discussions: use the GitHub repository Issues page.
- Example datasets: see /data/examples in the repo.

Setup

Requirements
- Python 3.10 or later.
- 8+ GB RAM for small sets. 16+ GB RAM recommended for larger models.
- Disk space for indexes and embeddings. Plan 1–10 GB per dataset.
- Optional GPU for model inference (CUDA 11.7+).
- curl, wget, tar for release installers.

Install from releases
1. Visit the Releases page and download the packaged release asset.
2. The release includes an installer script and demo assets. After download, extract and run the installer:
   - Linux / macOS
     - tar -xzf PersianRAG-release.tar.gz
     - cd PersianRAG-release
     - ./install.sh
   - Windows (WSL recommended)
     - unzip PersianRAG-release.zip
     - cd PersianRAG-release
     - .\install.bat
3. The installer will create a virtual environment, install Python dependencies, and place demo data in ./data.

Local installation (source)
- Create and activate a venv:
  - python -m venv .venv
  - source .venv/bin/activate
- Install from pip:
  - pip install -r requirements.txt
- Run initial setup:
  - python -m persianrag.setup --download-models

Docker
- Build:
  - docker build -t persianrag:latest .
- Run demo container:
  - docker run --rm -p 8080:8080 persianrag:latest
- The container exposes a demo web UI at http://localhost:8080.

Core concepts

Retrieval
- PersianRAG uses a retriever to find context from a document store.
- Use dense retrieval with embeddings and FAISS for semantic search.
- Use BM25 for efficient keyword matching on Persian text.
- Combine dense and sparse results with a reranker or MMR.

Augmented generation
- The generator is an LLM that produces answers using retrieved context.
- The system forms a prompt that contains the user query and the top-k retrieved passages.
- The model then outputs a concise answer and an optional provenance list.

Indexing and embeddings
- Tokenize Persian using UTF-8 and a simple word-segmentation suited for Persian.
- Use SentencePiece or fastText embeddings trained for Persian.
- Vector index uses FAISS with IVF or HNSW for large collections.
- Keep the embedding model and the generator separate. This gives flexibility.

Usage

Command-line demo
- Start a demo server:
  - python -m persianrag.server --port 8080
- Run a CLI query:
  - python -m persianrag.client --query "تاریخچه تلفن در ایران چیست؟"
- Index a folder:
  - python -m persianrag.index --source ./docs --name myindex

Python API example
- Minimal example to run a query:
  - from persianrag import Retriever, Generator
  - retriever = Retriever(index_path='indexes/myindex')
  - generator = Generator(model_name='local-persian-llm')
  - docs = retriever.retrieve('بهترین روش مطالعه چیست؟', top_k=5)
  - prompt = generator.build_prompt(query='بهترین روش مطالعه چیست؟', docs=docs)
  - answer = generator.generate(prompt)
  - print(answer)

Web UI
- The demo UI shows:
  - Query box for Persian input.
  - Result list with passages and scores.
  - Source links or document IDs for provenance.
- Customize UI templates in /web/templates.

Data preparation

Supported document types
- PDFs: Extract text with pdfminer or PyMuPDF.
- DOCX: Use python-docx to extract text.
- HTML: Use beautifulsoup to strip tags and retain text.
- Plain text: UTF-8 encoded files.

Document splitting
- Split documents into passages of 200–500 tokens.
- Keep overlap of 50–100 tokens to preserve context.
- Store passage metadata: doc_id, offset, filename.

Persian text tips
- Normalize Arabic letters (ی vs ي), and remove non-standard characters.
- Normalize diacritics if present.
- Handle zero-width joiners and non-breaking spaces.
- Use word segmentation tuned for Persian if using token-based sparse methods.

Models and components

Embedding models
- fastText Persian embeddings (good for sparse)
- Sentence-BERT models fine-tuned on Persian text for dense retrieval

Retrieval backends
- FAISS for dense vectors
- Elasticsearch + BM25 for hybrid setups
- HNSW for low-latency nearest neighbor search

Generators
- Local LLMs (if you host one)
- Hosted endpoints (for production with adequate latency)
- Template-based post-processing to format answers and citations

Prompt templates
- Use an instruction template with explicit constraints.
- Provide retrieved passages annotated with source IDs.
- Limit context length to match model context windows.

Performance and tuning

Indexing
- Use IVF + PQ for very large corpora.
- Use HNSW for low-latency queries on medium-sized corpora.

Retrieval accuracy
- Tune top_k and rerank thresholds.
- Combine BM25 and dense scores with a weighted sum.
- Use MMR for diversity when you need broader coverage.

Generator quality
- Control temperature and max tokens.
- Use system and user role separation in prompts.
- Use a safety filter for production use.

Testing and benchmarks

Local benchmarks
- Use a small Persian QA dataset for evaluation.
- Measure exact match (EM) and F1 on answers.
- Record average retrieval latency, generator latency, and end-to-end latency.

Sample benchmark results (example)
- Dataset: PersianQA-small (500 queries)
- Retriever recall@5: 0.86
- Generator F1: 0.72
- End-to-end latency: 600–1200 ms on CPU; 120–300 ms on GPU

The repo includes scripts to run these tests in /benchmarks.

Download and run release assets
- The Releases page hosts packaged builds and demo assets. Download the release asset and run the included installer or demo script from the extracted folder. Typical steps:
  - wget https://github.com/sheoranvijay776/PersianRAG/releases/download/v1.0/PersianRAG-release.tar.gz
  - tar -xzf PersianRAG-release.tar.gz
  - cd PersianRAG-release
  - ./run_demo.sh
- The release will contain a ready-made index, a small model or model adapters, and demo queries.

Deployment

Server mode
- Use the built-in FastAPI server for production.
- Run with a process manager like systemd or docker-compose.
- Set up TLS and a reverse proxy for public access.

Scaling
- Separate services: indexer, retriever, generator.
- Use a vector search cluster for large indexes.
- Cache frequent queries at the API layer.

Security
- Authenticate API access with tokens.
- Limit exposed endpoints when running in public mode.

Contribute
- Fork the repository and open a PR.
- Add tests under /tests and ensure they pass.
- Document new components in ./docs.
- Report bugs and request features via Issues.

Repository layout (summary)
- /persianrag — main package
- /data — sample datasets and index snapshots
- /docs — deeper guides and architecture notes
- /benchmarks — test scripts and sample results
- /web — demo UI and assets
- /scripts — helper scripts

FAQ

Q: Which retriever should I use?
A: Use FAISS for semantic search. Use BM25 for keyword search. Combine them for best recall.

Q: How do I add custom documents?
A: Place files in ./data/docs and run the index command:
- python -m persianrag.index --source ./data/docs --name myindex

Q: How do I tune prompt length?
A: Trim retrieved passages to fit the model context window. Use summary passaging for long docs.

Acknowledgements
- FAISS for fast vector search.
- Sentence-BERT and fastText communities for embedding models.
- Open-source PDF and doc parsing tools.

License
- This project uses the MIT License. See LICENSE.md for details.

Changelog and releases
- Check the Releases page for packaged builds, release notes, and downloadable assets: https://github.com/sheoranvijay776/PersianRAG/releases

Images and assets used
- Hero image from Unsplash (free images).
- Badges generated with img.shields.io.

How to get help
- Open an issue on the repository for bugs or feature requests.
- Use the Discussions tab for usage questions and design ideas.

Examples and common commands
- Index a directory:
  - python -m persianrag.index --source data/my-docs --name myindex
- Query the index:
  - python -m persianrag.query --index myindex --q "تاثیر محمود درباره"
- Run tests:
  - pytest tests

Contact
- Use the repository Issues page to report problems or ask for help.

References
- RAG papers and implementation notes
- FAISS documentation
- Persian NLP and tokenization guides

The README above gives a full workflow from install to deploy. The Releases page contains packaged artifacts and demo scripts. Download a release, run the included installer, and try the demo to see PersianRAG in action.