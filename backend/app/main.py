"""RAG Docs Chat API — chat with your documents.

FastAPI + Qdrant (local embeddings) + any OpenAI-compatible LLM (DeepSeek by default).
"""
from pathlib import Path

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.extractor import extract_text
from app.rag.chunker import chunk_text
from app.rag.llm import LLMClient, build_answer_messages
from app.rag.vectorstore import VectorStore

app = FastAPI(
    title="RAG Docs Chat",
    description="Chat with your documents: FastAPI + Qdrant (local embeddings) + any OpenAI-compatible LLM.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask your documents")
    top_k: int = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")


def get_vectorstore() -> VectorStore:
    return VectorStore(settings.qdrant_url, settings.qdrant_collection)


def get_llm() -> LLMClient:
    return LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/ingest", summary="Upload PDF/TXT/MD documents and index them")
def ingest(
    files: list[UploadFile] = File(...),
    vectorstore: VectorStore = Depends(get_vectorstore),
):
    chunks: list[dict] = []
    names: list[str] = []
    for file in files:
        raw = file.file.read()
        text = extract_text(file.filename or "document.txt", raw)
        for part in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append({"text": part, "source": file.filename or "document.txt"})
        names.append(file.filename or "document.txt")
    count = vectorstore.ingest(chunks)
    return {"status": "ok", "indexed_chunks": count, "files": names}


@app.post("/query", summary="Ask a question over the indexed documents")
def query(
    payload: QueryRequest,
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLMClient = Depends(get_llm),
):
    hits = vectorstore.search(payload.question, payload.top_k)
    if not hits:
        return {
            "answer": "No relevant documents found in the index. Ingest documents first.",
            "sources": [],
        }
    messages = build_answer_messages(payload.question, hits)
    try:
        answer = llm.complete(messages)
    except RuntimeError as exc:
        # Graceful degradation: without an LLM key the retrieved context is still returned.
        return {
            "answer": f"{exc} — here are the top relevant passages instead:",
            "sources": hits,
        }
    return {"answer": answer, "sources": hits}


# Serve the built frontend when available (optional — API-only works without it).
# Works both locally (backend/app/main.py) and in the Docker image (app/main.py).
def _find_frontend_dist() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = current / "frontend" / "dist"
        if candidate.is_dir():
            return candidate
        current = current.parent
    return None


_ui_dir = _find_frontend_dist()
if _ui_dir:
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
