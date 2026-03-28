# backend/app/models/conversation.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from backend.app.models.base import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id         = Column(Integer, primary_key=True, index=True)
    source     = Column(String(500), nullable=False)
    file_type  = Column(String(20))
    content    = Column(Text, nullable=False)
    embedding  = Column(Vector(384))
    created_at = Column(DateTime, server_default=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    messages   = relationship("ChatMessage", back_populates="session",
                              cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id"),
                        nullable=False)
    role       = Column(String(20), nullable=False)
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    session    = relationship("ChatSession", back_populates="messages")
