"""
Embedding model — loaded once at startup, reused across all requests.
Must be the same model used during ingestion and retrieval.
"""
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list[float]:
    """
    Convert a text string into a 384-dimensional vector.

    Args:
        text: any string — chunk content or user query

    Returns:
        list of 384 floats representing semantic meaning
    """
    return embedding_model.encode(text).tolist()