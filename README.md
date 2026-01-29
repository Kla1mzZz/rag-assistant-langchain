# AI Assistant with RAG

An intelligent AI assistant service built with FastAPI that implements Retrieval-Augmented Generation (RAG) using LangChain, LangGraph, Google Gemini and Qdrant. The service intelligently routes queries between general LLM responses and document-based RAG responses.

## Features

- 🤖 **Smart Query Routing**: Automatically determines whether a query requires document retrieval or can be answered by the general LLM
- 📚 **RAG Pipeline**: Retrieval-Augmented Generation for document-based question answering
- 📄 **Document Management**: Upload, list, and delete documents (PDF, DOCX, TXT)
- 🔍 **Query Optimization**: Automatically optimizes user queries for better retrieval
- 💬 **Conversational AI**: Maintains conversation context using LangGraph agents (per `thread_id`)
- 🚀 **FastAPI**: Modern, fast web framework with automatic API documentation
- 🐳 **Docker & docker-compose**: Containerized deployment with embedded Qdrant vector database and Redis cache
- ⚡ **Redis cache**: Caches responses, document list, and RAG results to speed up repeated requests

## Tech Stack

- **Framework**: FastAPI
- **LLM**: Google Gemini (via LangChain)
- **Orchestration**: LangGraph
- **Vector Store**: Qdrant (`langchain-qdrant` + `qdrant-client`)
- **Embeddings**: Sentence Transformers (multilingual-e5-base, 768 dimensions)
- **Document Processing**: LangChain document loaders
- **Cache**: Redis (optional, for response and RAG caching)
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

# Redis cache (optional; cache is disabled without Redis)
CACHE__ENABLED=true
CACHE__REDIS_URL=redis://redis:6379/0
CACHE__DOCUMENTS_TTL_SECONDS=300
CACHE__CONVERSATION_TTL_SECONDS=600
CACHE__RAG_RETRIEVE_TTL_SECONDS=600
CACHE__OPTIMIZE_QUERY_TTL_SECONDS=600
CACHE__GENERATE_TTL_SECONDS=600
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
- `redis` (Redis 7 Alpine) on port `6379` for caching (when `CACHE__ENABLED=true`)

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

## Redis: Caching

Redis is used as an optional cache to reduce load on the LLM and Qdrant and speed up repeated requests.

### What is cached

| Cache type        | Key prefix       | Default TTL | Description                                      |
|-------------------|------------------|-------------|--------------------------------------------------|
| **Documents list**| `documents:*`    | 5 min       | Result of GET `/documents`                       |
| **Query optimization** | `optimize_query:*` | 10 min  | RAG query rewrite (optimized query)             |
| **RAG retrieve**  | `rag_retrieve:*` | 10 min      | Reserved for vector search result cache          |
| **Generation response** | `generate:*` | 10 min  | LLM response for context + query                 |
| **Conversation**  | `conversation:*` | 10 min     | Conversation cache keys                          |

Keys are built from a SHA-256 hash of the normalized query/prompt text, so identical requests hit the same cache entry.

### Configuration

- **Enable/disable**: `CACHE__ENABLED=true|false`. If `false` or Redis is unavailable, caching is disabled and the app runs without it.
- **URL**: `CACHE__REDIS_URL` (e.g. `redis://localhost:6379/0` or in Docker `redis://redis:6379/0`).
- **TTL** (seconds): `CACHE__DOCUMENTS_TTL_SECONDS`, `CACHE__CONVERSATION_TTL_SECONDS`, `CACHE__RAG_RETRIEVE_TTL_SECONDS`, `CACHE__OPTIMIZE_QUERY_TTL_SECONDS`, `CACHE__GENERATE_TTL_SECONDS`.

When a document is added or deleted via the API, cache invalidation runs: `documents:*` and `rag_retrieve:*` keys are cleared so the document list and search results stay up to date.

## Vector Search: How It Works

Document search is built on the **Qdrant** vector store and embeddings. Below is how the pipeline is composed and what benefits it provides.

### Embeddings and collection

- **Model**: `intfloat/multilingual-e5-base` (Sentence Transformers) — multilingual, suitable for English and other languages.
- **Vector size**: 768 dimensions.
- **Distance metric**: cosine similarity (`COSINE`).
- **Qdrant collection**: `rag_store`; created automatically on first run with `VectorParams(size=768, distance=COSINE)`.

Text is split into chunks (see below); each chunk is embedded and stored in Qdrant with metadata.

### Chunks and metadata

- **Splitter**: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`.
- **Chunk size and overlap**: configured via `RAG__CHUNK_SIZE` and `RAG__CHUNK_OVERLAP` (defaults 1500 and 150).
- Metadata includes: `source`, `chunk_id`, `created_at`, `file_size` (and others as needed). Documents are deleted by `metadata.source` (all points with that source are removed).

Point IDs in Qdrant are deterministic UUIDs (uuid5) from `{source}_{content_hash}`, so re-indexing the same content does not create duplicates.

### Retrieval pipeline

1. **Base retriever**: Qdrant search in **MMR** (Max Marginal Relevance) mode:
   - `search_type="mmr"`;
   - `fetch_k=50` — number of candidates fetched from Qdrant;
   - `k=20` — number returned after MMR (balance of relevance and diversity).
2. **Reranker**: **FlashrankRerank** (contextual compression) is applied on those 20 documents. It reranks by relevance to the query and trims to top-**k** (default **k=4** in RAG).
3. **Result**: the user gets 4 most relevant and diverse chunks, improving context quality for the LLM and reducing noise.

Summary: vector search benefits in this project:

- **Semantic search** by meaning, not exact keyword match.
- **MMR** — fewer near-duplicate chunks, more diversity in context.
- **Rerank (Flashrank)** — more accurate top results after the initial retrieval.
- **Multilingual** — one model for multiple languages.
- **Metadata filtering** — delete by source (`metadata.source`); filters can be extended in Qdrant as needed.

## LangSmith monitoring

[LangSmith](https://smith.langchain.com/) provides observability for LangChain/LangGraph: traces of LLM calls, RAG retrieval, and graph execution.

1. Sign up at [smith.langchain.com](https://smith.langchain.com/) and create an API key.
2. Set in `.env`:
   - `LANGCHAIN__TRACING_V2=true` — enable tracing
   - `LANGCHAIN__API_KEY=<your_langsmith_api_key>`
   - `LANGCHAIN__PROJECT=ai-assistant` (or any project name in LangSmith)
3. Start the app; traces will appear in the LangSmith project for each conversation, gatekeeper decision, RAG retrieve, and LLM generation.

If `LANGCHAIN__API_KEY` is empty or `LANGCHAIN__TRACING_V2=false`, tracing is disabled and no data is sent.

## Configuration

Configuration is managed through environment variables with nested structure support:

- `LLM__*`: LLM configuration (model, temperature, API key)
- `RAG__*`: RAG pipeline configuration (embeddings, chunk size, Qdrant URL, etc.)
- `CACHE__*`: Redis cache (enabled, URL, TTL for documents, conversations, RAG, generation)
- `LANGCHAIN__*`: LangSmith tracing (tracing_v2, api_key, project)
- `APP__*`: Application configuration (host, port, debug mode)

See `src/ai_assistant/core/config.py` for all available options.

## Development

### Testing

Install test dependencies and run tests:

```bash
uv sync --extra test
uv run pytest tests -v
```

Tests cover configuration, cache key helpers, Pydantic schemas, the RAG splitter, and API endpoints (with mocked LLM/Qdrant). API tests use FastAPI's `TestClient` and mock `rag_graph` and `pipeline` where needed.

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