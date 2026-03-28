# backend/app/schemas/session.py
from pydantic import BaseModel

class ClearRequest(BaseModel):
    session_id: str

class SessionListResponse(BaseModel):
    sessions: list[str]