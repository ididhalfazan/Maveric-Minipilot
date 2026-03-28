# backend/app/schemas/chat.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    module_focus: str | None = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]