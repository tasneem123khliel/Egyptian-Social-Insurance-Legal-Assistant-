"""REST API for the Arabic Social Insurance RAG assistant (for Postman testing).

Run locally:  uvicorn api:app --host 0.0.0.0 --port 8000
Docs (Swagger): http://localhost:8000/docs

Endpoints:
  GET  /health  -> service + knowledge-base status
  POST /ask     -> {"question": "...", "top_k": 5} => {"answer": "...", "sources": [...]}
"""

import os
import sys
import threading

# Guarantee the sibling rag_core.py next to this file is the one imported,
# no matter which folder the terminal was opened in.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_core import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    ask_question,
    build_chain,
    load_and_split_pdf,
    load_or_build_vectorstore,
)

PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "law-79-1975.pdf")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

app = FastAPI(
    title="Arabic Social Insurance RAG API",
    description="اسأل بالعربية عن قانون التأمين الاجتماعي المصري — إجابة + مصادر بأرقام الصفحات",
    version="1.0.0",
)

_lock = threading.Lock()
_state = {"retriever": None, "chunks": 0}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="السؤال باللغة العربية")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=8, description="عدد المقاطع المسترجعة")


class Source(BaseModel):
    page: int
    text: str
    source: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


def _get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server (environment variable).",
        )
    return key


def _get_retriever():
    """Build the knowledge base lazily on first question (slow first call, fast after)."""
    if _state["retriever"] is not None:
        return _state["retriever"]
    with _lock:
        if _state["retriever"] is not None:
            return _state["retriever"]
        # Reuse an existing chroma_db folder when present (no rebuild, no duplicates).
        splits = None
        db_exists = os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR)
        if not db_exists:
            if not os.path.exists(PDF_PATH):
                raise HTTPException(status_code=500, detail=f"PDF not found: {PDF_PATH}")
            _, splits = load_and_split_pdf(PDF_PATH)
        vectorstore = load_or_build_vectorstore(
            splits, embedding_model=DEFAULT_EMBEDDING_MODEL, persist_dir=PERSIST_DIR
        )
        _state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 8})
        _state["chunks"] = vectorstore._collection.count()
        return _state["retriever"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "knowledge_base_loaded": _state["retriever"] is not None,
        "chunks": _state["chunks"],
        "llm_model": DEFAULT_LLM_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retriever = _get_retriever()
    chain = build_chain(_get_api_key(), llm_model=DEFAULT_LLM_MODEL)
    answer, sources = ask_question(chain, retriever, req.question, top_k=req.top_k)
    return AskResponse(question=req.question, answer=answer, sources=sources)
