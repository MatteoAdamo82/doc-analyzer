# Doc Analyzer

A web application that analyzes PDF, DOC, and DOCX documents using large language models through Ollama and RAG (Retrieval-Augmented Generation) architecture.

## Overview

Doc Analyzer enables users to:
- Upload PDF, DOC, and DOCX documents
- Ask questions about document content
- Receive AI-generated responses based on document content
- Process documents using state-of-the-art language models

The application leverages:
- Ollama for text generation and embeddings
- ChromaDB for vector storage
- LangChain for document processing
- FastAPI and Gradio for the web interface

## Project Structure

```
doc-analyzer/
├── src/                        # Source code
│   ├── app.py                  # Main FastAPI application
│   ├── config/                 # Configuration files
│   │   ├── __init__.py
│   │   └── prompts.py          # Role-based prompts configuration
│   └── processors/             # Document processors
│       ├── base/               # Base classes
│       │   └── document_processor.py
│       ├── factory.py          # Factory for processor creation
│       ├── pdf_processor.py    # PDF document handling
│       ├── word_processor.py   # Word document handling
│       └── rag_processor.py    # RAG implementation
├── tests/                      # Test files
│   ├── processors/             # Processor-specific tests
│   │   ├── test_base_processor.py
│   │   ├── test_factory.py
│   │   └── test_word_processor.py
│   └── unit/                   # Unit tests
│       ├── test_app.py
│       └── test_rag_processor.py
├── data/                       # Data directory
│   └── chroma/                 # ChromaDB storage
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
├── docker-compose.test.yml     # Test container configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── setup.py                    # Package setup
```

## Requirements

- Docker and Docker Compose
- Ollama with your preferred language model
- 8GB RAM minimum
- 20GB disk space
- Additional system dependencies (managed by Docker):
  - poppler-utils (for PDF processing)
  - tesseract-ocr and libtesseract-dev (for text extraction)
  - antiword and unrtf (for DOC/DOCX processing)

### Installing Ollama

1. Install Ollama:
   - Linux: `curl https://ollama.ai/install.sh | sh`
   - macOS/Windows: Download from https://ollama.ai

2. Pull the language model (deepseek-r1:14b is the default, but you can use any Ollama model by updating the .env file):
```bash
ollama pull deepseek-r1:14b  # or your preferred model
```

## Quick Start with Docker

1. Clone and setup:
```bash
git clone https://github.com/MatteoAdamo82/doc-analyzer
cd doc-analyzer
cp .env.example .env
```

2. Configure `.env`:
```env
OLLAMA_HOST=host.docker.internal  # Use 'localhost' for local dev
OLLAMA_PORT=11434
CHROMA_DB_PATH=/app/data/chroma
LLM_MODEL=mistral:7b  # Use your preferred Ollama model
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
PERSIST_VECTORDB=false
```

3. Running the application:
```bash
# Start the web application
docker compose up -d

# Access the web interface at http://localhost:8000
```

4. Running tests:
```bash
# Run the test suite
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

5. Other Docker commands:
```bash
# Clean start: remove containers from old configurations
docker-compose up --remove-orphans

# View logs
docker-compose logs -f

# Rebuild without using cached images
docker-compose build --no-cache

# Restart services
docker-compose restart
```

### Deployment Without Docker
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
mkdir -p ./data/chroma
chmod -R 777 ./data
uvicorn src.app:app --reload
```

## Usage Guide

1. **Upload And Process Document**
   - Click upload area or drag-and-drop your document
   - Supported formats: PDF, DOC, DOCX
   - Click Process Document button
   - Wait for processing completion

2. **Ask Questions**
   - Type your question in the text input
   - Select an analysis role from the dropdown:
     - Default: General document analysis
     - Legal: Legal implications and compliance analysis
     - Financial: Financial implications and economic analysis
     - Travel: Travel-related insights and recommendations
     - Technical: Technical details and implementation analysis
   - Click "Ask" or press Enter
   - Receive role-specific AI-generated responses

3. **Best Practices**
   - Use clear, specific questions
   - Ask one question at a time
   - For complex documents, start with general questions
   - Wait for each response before asking next question

## Architecture

The application consists of several components:

### Web Interface (FastAPI + Gradio)
- Handles file uploads and user interactions
- Provides intuitive interface
- Manages user sessions

### Document Processors
- PDF Processor:
  - Extracts text from PDF documents using PyMuPDF
  - Splits content into manageable chunks
  - Uses LangChain for document handling
- Word Processor:
  - Processes DOC files using antiword
  - Handles DOCX files using python-docx
  - Extracts text content for analysis

### RAG Processor
- Creates embeddings using DeepSeek
- Stores vectors in ChromaDB
- Retrieves relevant content for queries
- Generates responses using DeepSeek

### Vector Store (ChromaDB)
- Stores document embeddings
- Enables semantic search
- Maintains document-query relevance

## Troubleshooting

### Database Issues
If you get "readonly database" error in local development:
```bash
chmod -R 777 ./data
rm -rf ./data/chroma/*
```

### Ollama Connection
1. Verify Ollama is running: `ollama list`
2. Check service: `curl http://localhost:11434/api/tags`
3. Check `.env` configuration matches your setup
4. For Docker Desktop users, ensure `host.docker.internal` is used

### Memory Issues
- Docker Desktop: Increase memory limit (Preferences → Resources → Memory)
- Recommended minimum: 8GB
- If needed, reduce chunk size in `pdf_processor.py`:
```python
chunk_size=500  # Decrease if experiencing memory issues
chunk_overlap=100
```

### Document Processing Issues
- PDF files: Ensure the PDF is not password-protected
- DOC files: File must be readable by antiword
- DOCX files: File must be a valid Office Open XML format
- If text extraction fails, try converting the document to PDF

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits focused and clean

## License

MIT License - see LICENSE file for details.