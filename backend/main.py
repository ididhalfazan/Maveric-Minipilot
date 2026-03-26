"""
Maveric MiniPilot — FastAPI Backend
====================================
REST API that orchestrates the RAG pipeline and conversation management.

Endpoints:
    GET  /health    — liveness check
    POST /chat      — main chat endpoint
    POST /clear     — delete a session
    GET  /sessions  — list all sessions

Dependencies:
    - Groq API      : LLM answer generation
    - PostgreSQL    : conversation persistence
    - RAG engine    : document retrieval
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy.orm import Session

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.connection import get_db, init_db
from database.models import ChatSession, ChatMessage
from rag_engine.retriever import retrieve

app = FastAPI(title="Maveric MiniPilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Maveric MiniPilot, an AI assistant that guides users through
the Maveric RIC Algorithm Development Platform.

You help users understand:
- Full workflow: simulation setup, UE tracks generation, Digital Twin training,
  RF Prediction, and job orchestration
- Module deep-dives: Digital Twin, RF Prediction, UE Tracks Generation, Orchestration
- Inputs, outputs, internal flows, and microservice references for each module

Always base your answers on the provided documentation context.
If the context does not contain enough information, say so honestly.
Be concise, clear, and guide the user step by step."""

MAVERIC_MODULES = {
    "digital_twin":  "Digital Twin module",
    "rf_prediction": "RF Prediction module",
    "ue_tracks":     "UE Tracks Generation module",
    "orchestration": "Orchestration / Job Orchestration module",
    "workflow":      "full Maveric end-to-end workflow",
}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    module_focus: str | None = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]

class ClearRequest(BaseModel):
    session_id: str

@app.on_event("startup")
def on_startup():
    init_db()

def get_or_create_session(session_id: str, db: Session) -> ChatSession:
    session = db.query(ChatSession).filter_by(session_id=session_id).first()
    if not session:
        session = ChatSession(session_id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session

def get_history(session_id: str, db: Session, limit: int = 10) -> list[dict]:
    messages = (
        db.query(ChatMessage)
        .filter_by(session_id=session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]

def save_messages(session_id: str, user_msg: str, assistant_msg: str, db: Session):
    db.add(ChatMessage(session_id=session_id, role="user",      content=user_msg))
    db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_msg))
    db.commit()

@app.get("/")
def root():
    return {"status": "Maveric MiniPilot running", "modules": list(MAVERIC_MODULES.keys())}

@app.get("/health")
def health(db: Session = Depends(get_db)):
    session_count = db.query(ChatSession).count()
    return {"status": "ok", "sessions_in_db": session_count}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint — full RAG + LLM pipeline.

    Pipeline:
        1. Optionally prefix query with module context
        2. Retrieve top-4 chunks from pgvector
        3. Load conversation history from PostgreSQL
        4. Assemble prompt: system + context + history + question
        5. Call Groq LLaMA3 for answer
        6. Save message pair to PostgreSQL
        7. Return answer + source file paths

    Args:
        req: ChatRequest with message, session_id, optional module_focus
        db:  injected PostgreSQL session

    Returns:
        ChatResponse with answer, session_id, and source file list

    Raises:
        400: empty message
        500: RAG retrieval or Groq API failure
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    query = req.message
    if req.module_focus and req.module_focus in MAVERIC_MODULES:
        query = f"{MAVERIC_MODULES[req.module_focus]}: {req.message}"

    try:
        chunks = retrieve(query, k=4, db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {str(e)}")

    context_parts = []
    sources = []
    for chunk in chunks:
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['content']}")
        if chunk["source"] not in sources:
            sources.append(chunk["source"])
    context = "\n\n---\n\n".join(context_parts)

    get_or_create_session(req.session_id, db)
    history = get_history(req.session_id, db)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Relevant Maveric documentation:\n\n{context}"},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq call failed: {str(e)}")

    save_messages(req.session_id, req.message, answer, db)

    return ChatResponse(answer=answer, session_id=req.session_id, sources=sources)

@app.post("/clear")
def clear_session(req: ClearRequest, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter_by(session_id=req.session_id).first()
    if session:
        db.delete(session)
        db.commit()
    return {"cleared": req.session_id}

@app.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).all()
    return {"sessions": [s.session_id for s in sessions]}
