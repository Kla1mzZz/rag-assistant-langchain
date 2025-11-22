# AI Assistant

An intelligent AI assistant service built with FastAPI that implements Retrieval-Augmented Generation (RAG) using LangChain, LangGraph, and Google Gemini. The service intelligently routes queries between general LLM responses and document-based RAG responses.

## Features

- 🤖 **Smart Query Routing**: Automatically determines whether a query requires document retrieval or can be answered by the general LLM
- 📚 **RAG Pipeline**: Retrieval-Augmented Generation for document-based question answering
- 📄 **Document Management**: Upload, list, and delete documents (PDF, DOCX, TXT)
- 🔍 **Query Optimization**: Automatically optimizes user queries for better retrieval
- 💬 **Conversational AI**: Maintains conversation context using LangGraph agents
- 🚀 **FastAPI**: Modern, fast web framework with automatic API documentation
- 🐳 **Docker Support**: Containerized deployment ready

## Tech Stack

- **Framework**: FastAPI
- **LLM**: Google Gemini (via LangChain)
- **Orchestration**: LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence Transformers (multilingual-e5-base)
- **Document Processing**: LangChain document loaders
- **Python**: 3.12+

## Project Structure

```
ai-assistant/
├── src/
│   └── ai_assistant/
│       ├── api/              # API endpoints
│       │   ├── v1/
│       │   │   ├── chat.py   # Conversation endpoints
│       │   │   └── admin.py  # Document management
│       │   └── health.py     # Health check
│       ├── core/             # Core configuration and utilities
│       │   ├── config.py     # Application configuration
│       │   └── logger.py     # Logging setup
│       ├── graph/            # LangGraph orchestration
│       │   ├── graph.py      # RAG graph definition
│       │   └── state.py      # State management
│       ├── rag/              # RAG pipeline components
│       │   ├── pipeline.py   # Main RAG pipeline
│       │   ├── embeddings.py # Embedding models
│       │   ├── splitter.py   # Document splitting
│       │   └── vector_store.py # Vector store management
│       ├── schemas/          # Pydantic models
│       ├── utils/            # Utility functions
│       └── main.py           # FastAPI application entry point
├── docs/                     # Document storage
├── prompts/                  # Prompt templates
├── chromadb/                 # ChromaDB persistence
├── pyproject.toml           # Poetry dependencies
├── Dockerfile               # Docker configuration
└── README.md
```

## Installation

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-assistant
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root:
```env
# Google Gemini API Key
LLM__API_KEY=your_google_gemini_api_key_here

# Optional: Override default settings
LLM__MODEL_NAME=gemini-2.5-flash
LLM__TEMPERATURE=0.75
LLM__TOP_K=50
LLM__TOP_P=0.9

# RAG Configuration
RAG__EMBEDDING_MODEL=intfloat/multilingual-e5-base
RAG__CHUNK_SIZE=800
RAG__CHUNK_OVERLAP=150

# App Configuration
APP__HOST=0.0.0.0
APP__PORT=8000
APP__DEBUG=false
```

4. Activate the virtual environment:
```bash
poetry shell
```

## Usage

### Running the Application

Start the FastAPI server:

```bash
poetry run uvicorn src.ai_assistant.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Docker

Build and run using Docker:

```bash
docker build -t ai-assistant .
docker run -p 8000:8000 --env-file .env ai-assistant
```

## API Endpoints

### Chat Endpoints

#### POST `/api/v1/conversation`
Start a conversation with the AI assistant.

**Request:**
```json
{
  "prompt": "What is the company's mission?"
}
```

**Response:**
```json
{
  "answer": "Based on the documents...",
  "document_sources": ["about_company.pdf", "ceo.txt"]
}
```

#### POST `/api/v1/conversation/stream`
Stream conversation responses (coming soon).

### Admin Endpoints

#### POST `/api/v1/documents`
Upload a document for indexing.

**Request:** Multipart form data with `file` field

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf"
}
```

#### GET `/api/v1/documents?limit=5`
List indexed documents.

**Response:**
```json
{
  "documents": ["document1.pdf", "document2.txt"]
}
```

#### DELETE `/api/v1/documents/{doc_name}`
Delete a document from the index.

**Response:**
```json
{
  "success": true,
  "deleted": "document.pdf"
}
```

### Health Check

#### GET `/health`
Check service health status.

## How It Works

1. **Query Reception**: User sends a query to the `/conversation` endpoint
2. **Gatekeeper Decision**: The LLM gatekeeper determines if the query requires document retrieval
3. **Routing**:
   - **No RAG**: Query is sent directly to the general LLM agent
   - **Use RAG**: Query is optimized, then relevant documents are retrieved
4. **Document Retrieval**: Similar documents are retrieved from ChromaDB using semantic search
5. **Response Generation**: The LLM generates a response using the retrieved context
6. **Response**: The answer and document sources are returned to the user

## Configuration

Configuration is managed through environment variables with nested structure support:

- `LLM__*`: LLM configuration (model, temperature, API key)
- `RAG__*`: RAG pipeline configuration (embeddings, chunk size, etc.)
- `APP__*`: Application configuration (host, port, debug mode)

See `src/ai_assistant/core/config.py` for all available options.

## Development

### Code Quality

The project uses `ruff` for linting and formatting:

```bash
poetry run ruff check .
poetry run ruff format .
```

### Adding Documents

Place documents in the `docs/` folder or upload them via the API. Supported formats:
- PDF (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)

## License

[Add your license here]

## Author

Kla1mzZ (kla1mzz16@gmail.com)

