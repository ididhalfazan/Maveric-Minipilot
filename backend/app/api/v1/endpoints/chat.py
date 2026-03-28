from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_service import run_chat_pipeline

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Main chat endpoint — delegates entirely to chat_service."""
    if not req.message.strip():
        raise HTTPException(status_code=400,
                            detail="Message cannot be empty")
    try:
        result = run_chat_pipeline(
            message=req.message,
            session_id=req.session_id,
            module_focus=req.module_focus,
            db=db
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=result["answer"],
        session_id=req.session_id,
        sources=result["sources"]
    )