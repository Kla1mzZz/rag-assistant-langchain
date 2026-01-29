# AI Assistant with RAG

An intelligent AI assistant service built with FastAPI that implements Retrieval-Augmented Generation (RAG) using LangChain, LangGraph, Google Gemini and Qdrant. The service intelligently routes queries between general LLM responses and document-based RAG responses.

## Features

- 🤖 **Smart Query Routing**: Automatically determines whether a query requires document retrieval or can be answered by the general LLM
- 📚 **RAG Pipeline**: Retrieval-Augmented Generation for document-based question answering
- 📄 **Document Management**: Upload, list, and delete documents (PDF, DOCX, TXT)
- 🔍 **Query Optimization**: Automatically optimizes user queries for better retrieval
- 💬 **Conversational AI**: Maintains conversation context using LangGraph agents (per `thread_id`)
- 🚀 **FastAPI**: Modern, fast web framework with automatic API documentation
- 🐳 **Docker & docker-compose**: Containerized deployment with embedded Qdrant vector database

## Tech Stack

- **Framework**: FastAPI
- **LLM**: Google Gemini (via LangChain)
- **Orchestration**: LangGraph
- **Vector Store**: Qdrant (`langchain-qdrant` + `qdrant-client`)
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
│       │   ├── config.py     # Application configuration (Pydantic Settings)
│       │   └── logger.py     # Logging setup
│       ├── graph/            # LangGraph orchestration
│       │   ├── graph.py      # RAG graph definition
│       │   └── state.py      # State management
│       ├── rag/              # RAG pipeline components
│       │   ├── pipeline.py   # Main RAG pipeline
│       │   ├── embeddings.py # Embedding models
│       │   ├── splitter.py   # Document splitting
│       │   └── vector_store.py # Qdrant vector store management
│       ├── schemas/          # Pydantic models
│       ├── utils/            # Utility functions
│       └── main.py           # FastAPI application entry point
├── docs/                     # Local document storage
├── prompts/                  # Prompt templates
├── qdrant_storage/           # Qdrant persistence (mounted in Docker)
├── pyproject.toml            # Project metadata & dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yaml       # App + Qdrant services
└── README.md
```

## Installation

### Prerequisites

- Python 3.12 or higher
- `pip` (or any PEP 621–compatible installer)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Kla1mzZz/rag-assistant-langchain
cd ai-assistant
```

2. (Recommended) Create and activate a virtual environment:

```bash
uv venv
# Windows
.venv\Scripts\activate
```

3. Install dependencies (from `pyproject.toml`):

```bash
uv sync
```

4. Create a `.env` file in the project root (you can use `.env.example` as a template):

```env
ENV=dev

# App Configuration
APP__TITLE=LLM Service
APP__VERSION=1.0.0
APP__DEBUG=true
APP__HOST=0.0.0.0
APP__PORT=8000

APP__CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
APP__CORS_HEADERS=["*"]
APP__CORS_CREDENTIALS=true

# Google Gemini API Key
LLM__API_KEY=your_google_gemini_api_key_here

# Optional: Override default LLM settings
LLM__MODEL_NAME=gemini-2.5-flash
LLM__TEMPERATURE=0.75
LLM__TOP_K=50
LLM__TOP_P=0.9

# RAG / Qdrant configuration
RAG__DB_URL=http://qdrant_db:6333
RAG__EMBEDDING_MODEL=intfloat/multilingual-e5-base
RAG__CHUNK_SIZE=800
RAG__CHUNK_OVERLAP=150
```

## Usage

### Running the Application

Start the FastAPI server:

```bash
uv run uvicorn src.ai_assistant.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Docker / docker-compose

Run the application together with Qdrant:

```bash
docker compose up --build
```

This will start:
- `app` service on port `8000`
- `qdrant_db` (Qdrant) on ports `6333`/`6334` with data in `qdrant_storage/`

You can still run the app container alone if you manage Qdrant separately:

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
  "prompt": "What is the company's mission?",
  "thread_id": "user-session-123"
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
List indexed documents with metadata.

**Response:**

```json
{
  "documents": [
    {
      "source": "document1.pdf",
      "date": "2025-01-01T12:00:00Z",
      "size": 12345.67 # MB
    }
  ]
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
4. **Document Retrieval**: Similar documents are retrieved from Qdrant using semantic search
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
ruff check .
ruff format .
```

### Adding Documents

Place documents in the `docs/` folder or upload them via the API. Supported formats:
- PDF (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)

## Author

Kla1mzZ (kla1mzz16@gmail.com)