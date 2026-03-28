# backend/app/services/chat_service.py
from groq import Groq
from sqlalchemy.orm import Session
from backend.app.core.config import settings
from backend.app.services.session_service import (
    get_or_create_session, get_history, save_messages
)
from backend.app.rag.retriever import retrieve

client = Groq(api_key=settings.GROQ_API_KEY)

def run_chat_pipeline(message: str, session_id: str,
                      module_focus: str | None,
                      db: Session) -> dict:
    """
    Full RAG + LLM pipeline.
    Returns dict with answer and sources.
    """
    # Prefix query with module context if selected
    query = message
    if module_focus and module_focus in settings.MAVERIC_MODULES:
        query = f"{settings.MAVERIC_MODULES[module_focus]}: {message}"

    # Retrieve relevant chunks above threshold
    chunks = retrieve(query, k=settings.RETRIEVAL_K, db=db,
                      threshold=settings.SIMILARITY_THRESHOLD)

    # Build context — honest fallback if nothing passes threshold
    if not chunks:
        context = (
            "No relevant documentation was found in the Maveric "
            "repository for this query. Answer from general knowledge "
            "if possible, or ask the user to rephrase."
        )
        sources = []
    else:
        parts, sources = [], []
        for chunk in chunks:
            parts.append(
                f"[Source: {chunk['source']} | "
                f"score: {chunk['score']:.3f}]\n{chunk['content']}"
            )
            if chunk["source"] not in sources:
                sources.append(chunk["source"])
        context = "\n\n---\n\n".join(parts)

    # Load history
    get_or_create_session(session_id, db)
    history = get_history(session_id, db)

    # Assemble prompt
    messages = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        {"role": "system",
         "content": f"Relevant Maveric documentation:\n\n{context}"},
        *history,
        {"role": "user", "content": message},
    ]

    # Call Groq
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800
    )
    answer = response.choices[0].message.content

    # Persist
    save_messages(session_id, message, answer, db)

    return {"answer": answer, "sources": sources}