# backend/app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.conversation import ChatSession
from backend.app.schemas.health import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    return HealthResponse(
        status="ok",
        sessions_in_db=db.query(ChatSession).count()
    )

@router.get("/")
def root():
    return {"status": "Maveric MiniPilot running"}