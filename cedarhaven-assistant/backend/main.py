import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_engine import CedarhavenRAGEngine

load_dotenv()

app = FastAPI(
    title="Cedarhaven Municipal AI Knowledge Assistant API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
rag_engine: Optional[CedarhavenRAGEngine] = None


@app.on_event("startup")
def startup_event():
    global rag_engine
    try:
        rag_engine = CedarhavenRAGEngine(data_dir=DATA_DIR)
        print(f"Loaded {len(rag_engine.chunks)} PDF pages across Cedarhaven documents.")
    except Exception as e:
        print(f"Warning during startup PDF ingestion: {e}")


class ChatRequest(BaseModel):
    question: str
    # Language selected in the frontend dropdown (English, French, German, Spanish, Mandarin, ...)
    language: Optional[str] = "English"


@app.get("/api/health")
def health_check():
    if rag_engine is None:
        return {"status": "warning", "chunks_loaded": 0}
    return {
        "status": "ok",
        "chunks_loaded": len(rag_engine.chunks),
        "documents": sorted({c.document_name for c in rag_engine.chunks}),
    }


@app.get("/api/evidence")
def evidence_endpoint(doc_id: str, page: int, q: str = ""):
    """Called only when the resident clicks a citation badge in the chat."""
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = CedarhavenRAGEngine(data_dir=DATA_DIR)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Knowledge base initialization failed: {str(e)}",
            )
    evidence = rag_engine.get_evidence(doc_id, page, q)
    if evidence is None:
        raise HTTPException(status_code=404, detail="That document page was not found.")
    return evidence


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="Please enter a valid non-empty question."
        )
    if len(question) > 1500:
        raise HTTPException(
            status_code=400,
            detail="Question is too long (maximum 1500 characters allowed).",
        )

    language = (req.language or "English").strip()[:40] or "English"

    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = CedarhavenRAGEngine(data_dir=DATA_DIR)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Knowledge base initialization failed: {str(e)}",
            )

    try:
        # Returns exactly {"answer": "...", "cited_pages": [{"doc_id": ..., "page_number": ...}]}
        return rag_engine.answer_question(question, language)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}",
        )