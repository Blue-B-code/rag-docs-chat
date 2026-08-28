# 📚 RAG Docs Chat — Chat with your documents

A self-contained **Retrieval-Augmented Generation (RAG)** app: upload PDF/TXT/Markdown files, then ask questions in natural language. Answers are grounded in your documents, with cited sources.

**FastAPI + Qdrant (local embeddings) + any OpenAI-compatible LLM (DeepSeek by default) + React frontend.**

---

## Overview

RAG Docs Chat lets you index a set of documents and ask questions about them. It combines three proven pieces:

1. **Ingestion** — files are parsed to text and split into overlapping chunks.
2. **Retrieval** — each chunk is embedded locally (no API key needed) and stored in **Qdrant**; your question is embedded the same way and the most relevant chunks are fetched.
3. **Generation** — a **large language model** (any OpenAI-compatible endpoint) writes a grounded answer, citing the retrieved chunks as `[1]`, `[2]`, …

## Problem

Knowledge lives in scattered documents — policies, handbooks, technical docs, PDFs. Searching them keyword-by-keyword is slow, and asking an LLM directly is unreliable because it has no access to your content (and may hallucinate).

## Solution

A small, transparent RAG pipeline that gives the LLM **only** the relevant passages from your own documents before it answers. Every answer comes with its sources, so the result is auditable.

## Key features

- **Local embeddings** with `fastembed` (`bge-small-en-v1.5`) — indexing works with **no API key**.
- **Qdrant** vector search (cosine similarity) with **Docker** support.
- **OpenAI-compatible LLM** — works with DeepSeek, OpenAI, Ollama, etc. via environment variables.
- **PDF / TXT / Markdown** ingestion with `pdfplumber`.
- **Graceful degradation** — without an LLM key, `/query` still returns the retrieved passages.
- **12 pytest tests** (chunking, prompt building, API) — offline, no key or network required.
- **Minimal React UI** served by the same FastAPI process (optional).

## Architecture

```
                        ┌─────────────────────────────┐
  Upload PDF/TXT/MD ───▶│  POST /ingest               │
                        │  extract_text → chunk_text  │
                        └──────────────┬──────────────┘
                                       ▼  embeddings (fastembed, local)
                        ┌─────────────────────────────┐
                        │  Qdrant (vector DB)         │
                        └──────────────┬──────────────┘
                                       ▲  query_text embeddings
                        ┌──────────────┴──────────────┐
  Question ────────────▶│  POST /query                │
                        │  retrieve top-k → prompt    │
                        │  → LLM (OpenAI-compatible)  │
                        └──────────────┬──────────────┘
                                       ▼
                        Answer + cited sources [1], [2], …
```

```
repo/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI routes: /health, /ingest, /query
│   │   ├── config.py          # Environment-driven settings
│   │   ├── extractor.py       # PDF / TXT / MD text extraction
│   │   └── rag/
│   │       ├── chunker.py     # Overlapping text chunking
│   │       ├── vectorstore.py # Qdrant + local fastembed embeddings
│   │       └── llm.py         # OpenAI-compatible chat client
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                  # Minimal React + Vite UI (built into dist/)
├── tests/                     # pytest (offline)
├── sample_docs/               # Demo document
├── docker-compose.yml         # Qdrant + API
└── .env.example
```

## Tech stack

- **Backend**: Python 3.12, FastAPI, Pydantic
- **Vector search**: Qdrant, fastembed (local embeddings)
- **LLM**: any OpenAI-compatible `/chat/completions` endpoint (DeepSeek by default)
- **Frontend**: React 18 + Vite
- **Testing**: pytest · **Infra**: Docker, docker-compose

## Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt

pytest -q
```

12 tests cover chunking (limits, overlap, word preservation), prompt building with citations, the LLM client (mocked transport) and the API (ingest, query, validation). **All offline** — no API key, no network.

## Docker / deployment

```bash
# 1. configure your LLM key
cp .env.example .env
#    then set LLM_API_KEY (e.g. your DeepSeek API key)

# 2. start Qdrant + API (builds the frontend too)
docker-compose up --build
```

- UI: http://localhost:8000
- Swagger API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

## Installation (local dev)

```bash
git clone https://github.com/Blue-B-code/rag-docs-chat.git
cd rag-docs-chat

python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt

# Terminal 1: Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Terminal 2: API
uvicorn app.main:app --app-dir backend --reload --port 8000

# Terminal 3 (optional): frontend dev server
cd frontend && npm install && npm run dev
```

## Usage

### Index documents

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@sample_docs/company-handbook.md"
# → {"status":"ok","indexed_chunks":3,"files":["company-handbook.md"]}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","top_k":3}'
```

Response includes the `answer` plus `sources` (text, filename, similarity score) for auditability.

### Configuration (`.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `LLM_API_KEY` | *(empty)* | Your LLM API key (DeepSeek, OpenAI, …) |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible base URL |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_MAX_TOKENS` | `512` | Max answer length |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Ingestion chunking |

## Screenshots

> _Add a screenshot of the UI and/or the Swagger docs here._

## Future improvements

- **Configurable embedding model** and multilingual models
- **Document deletion / re-ingestion** with the web UI
- **Metadata filters** (folder, date, tags) at query time
- **Streaming answers** (SSE)
- **Authentication** for hosted deployments

## License

MIT
