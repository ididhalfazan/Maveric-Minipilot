# backend/app/api/v1/endpoints/sessions.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.schemas.session import ClearRequest, SessionListResponse
from backend.app.services.session_service import (
    delete_session, list_all_sessions
)

router = APIRouter()

@router.post("/clear")
def clear_session(req: ClearRequest, db: Session = Depends(get_db)):
    delete_session(req.session_id, db)
    return {"cleared": req.session_id}

@router.get("/sessions", response_model=SessionListResponse)
def get_sessions(db: Session = Depends(get_db)):
    return SessionListResponse(sessions=list_all_sessions(db))