# backend/app/services/session_service.py
from sqlalchemy.orm import Session
from backend.app.models.conversation import ChatSession, ChatMessage

def get_or_create_session(session_id: str, db: Session) -> ChatSession:
    session = db.query(ChatSession).filter_by(session_id=session_id).first()
    if not session:
        session = ChatSession(session_id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session

def get_history(session_id: str, db: Session,
                limit: int = 10) -> list[dict]:
    messages = (
        db.query(ChatMessage)
        .filter_by(session_id=session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return [{"role": m.role, "content": m.content}
            for m in reversed(messages)]

def save_messages(session_id: str, user_msg: str,
                  assistant_msg: str, db: Session) -> None:
    db.add(ChatMessage(session_id=session_id,
                       role="user", content=user_msg))
    db.add(ChatMessage(session_id=session_id,
                       role="assistant", content=assistant_msg))
    db.commit()

def delete_session(session_id: str, db: Session) -> bool:
    session = db.query(ChatSession).filter_by(
        session_id=session_id).first()
    if session:
        db.delete(session)
        db.commit()
        return True
    return False

def list_all_sessions(db: Session) -> list[str]:
    return [s.session_id for s in db.query(ChatSession).all()]