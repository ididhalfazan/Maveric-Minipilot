# backend/app/schemas/health.py
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    sessions_in_db: int