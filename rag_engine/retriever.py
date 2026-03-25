"""
rag_engine/retriever.py
=======================
Semantic retrieval module for the Maveric MiniPilot RAG system.

This module is responsible for finding the most relevant documentation chunks
from the PostgreSQL vector store at query time. Given a user's question, it
generates an embedding using the same local sentence-transformers model used
during ingestion, then performs a cosine similarity search against all stored
chunk embeddings using the pgvector extension.

The top-k most similar chunks are returned as structured dictionaries containing
the source file path, raw text content, and similarity score. These chunks are
then injected into the LLM prompt in backend/main.py as grounding context,
enabling the Groq LLaMA3 model to answer questions based on actual Maveric
repository content rather than general training knowledge.

Retrieval Strategy:
    Method    : Cosine similarity via pgvector's <=> operator
    Model     : all-MiniLM-L6-v2 (same model used during ingestion)
    Default k : 4 chunks per query
    SQL       : ORDER BY embedding <=> query_vector LIMIT k

    Cosine similarity measures the angle between two vectors in the 384-
    dimensional embedding space. Chunks whose meaning is most similar to
    the query will have the smallest cosine distance (closest to 0), and
    therefore the highest similarity score (closest to 1.0).

    The similarity score returned is: 1 - cosine_distance
    where 1.0 = identical meaning, 0.0 = completely unrelated.

Consistency Requirement:
    The embedding model used here MUST be identical to the one used in
    ingest.py. If the models differ, the query vector and stored vectors
    will exist in different embedding spaces and similarity scores will
    be meaningless. Both files use 'all-MiniLM-L6-v2' producing 384-dim vectors.

Database Session:
    This module does not manage its own database session. A SQLAlchemy
    Session object must be passed in via the `db` parameter. In the FastAPI
    backend, this is provided automatically by the get_db() dependency injector
    defined in database/connection.py, ensuring proper session lifecycle
    management (one session per HTTP request, closed after response).

Usage:
    Called by backend/main.py on every /chat request:
        chunks = retrieve(query="What is the Digital Twin?", k=4, db=db_session)

Dependencies:
    - sentence-transformers : local embedding model (must match ingest.py)
    - SQLAlchemy            : database session and raw SQL execution
    - pgvector              : <=> cosine distance operator in PostgreSQL
    - python-dotenv         : loads .env config file
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(query: str) -> list[float]:
    """
    Generate a 384-dimensional embedding vector for a query string.

    Encodes the input using the locally loaded 'all-MiniLM-L6-v2' model.
    The output numpy array is converted to a plain Python list of floats
    for use in SQL parameter binding with the pgvector CAST syntax.

    This function is called once per user message in the retrieve() function
    below. The resulting vector is compared against all stored document
    chunk embeddings in PostgreSQL to find the most semantically similar chunks.

    Args:
        query (str):
            The user's question or search query. Can be a natural language
            question, a keyword phrase, or a technical term. The model handles
            all of these gracefully.
            Example: "How does RF prediction work in Maveric?"

    Returns:
        list[float]:
            A list of 384 floating point values representing the query's
            position in the model's embedding space. Values typically range
            between -1.0 and 1.0.

    Example:
        >>> vec = get_embedding("What is UE Tracks Generation?")
        >>> len(vec)
        384
        >>> all(isinstance(v, float) for v in vec)
        True
    """
    return embedding_model.encode(query).tolist()

def retrieve(query: str, k: int = 4, db=None) -> list[dict]:
    embedding = get_embedding(query)

    sql = text("""
        SELECT
            source,
            content,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :k
    """)

    result = db.execute(sql, {
        "embedding": str(embedding),
        "k": k
    })

    rows = result.fetchall()
    return [
        {
            "source": row.source,
            "content": row.content,
            "score": float(row.similarity)
        }
        for row in rows
    ]