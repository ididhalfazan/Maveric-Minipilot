"""
Retrieves top-k semantically relevant chunks from PostgreSQL
using pgvector cosine similarity search.
Includes similarity threshold to prevent unrelated chunk mixing.
"""
from sqlalchemy import text
from backend.app.rag.embeddings import get_embedding
from backend.app.core.config import settings

def retrieve(
    query: str,
    k: int = 4,
    db=None,
    threshold: float = None,
    source_filter: str = None,
    fetch_multiplier: int = 2
) -> list[dict]:
    """
    Find top-k relevant chunks for a query.

    Args:
        query:            user question or prefixed query string
        k:                number of chunks to return
        db:               active SQLAlchemy session
        threshold:        minimum similarity score — defaults to
                          settings.SIMILARITY_THRESHOLD if not passed
        source_filter:    restrict search to chunks whose source
                          path starts with this string
        fetch_multiplier: fetch k*multiplier before filtering
                          for better precision

    Returns:
        list of dicts with source, content, score
        empty list if nothing passes the threshold
    """
    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD

    embedding = get_embedding(query)
    fetch_k = k * fetch_multiplier

    if source_filter:
        sql = text("""
            SELECT
                source,
                content,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM document_chunks
            WHERE source LIKE :source_filter
            AND   1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :k
        """)
        result = db.execute(sql, {
            "embedding": str(embedding),
            "k": fetch_k,
            "threshold": threshold,
            "source_filter": f"{source_filter}%"
        })
    else:
        sql = text("""
            SELECT
                source,
                content,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM document_chunks
            WHERE 1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :k
        """)
        result = db.execute(sql, {
            "embedding": str(embedding),
            "k": fetch_k,
            "threshold": threshold,
        })

    rows = result.fetchall()

    chunks = [
        {
            "source": row.source,
            "content": row.content,
            "score": float(row.similarity)
        }
        for row in rows
    ]

    # Re-rank by score and return top k
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:k]