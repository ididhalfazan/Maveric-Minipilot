# backend/app/api/v1/router.py
from fastapi import APIRouter
from backend.app.api.v1.endpoints import chat, sessions, health

router = APIRouter()

router.include_router(health.router,   tags=["health"])
router.include_router(chat.router,     tags=["chat"])
router.include_router(sessions.router, tags=["sessions"])