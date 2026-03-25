"""
rag_engine/ingest.py
====================
Ingestion pipeline for the Maveric MiniPilot RAG system.

This module is responsible for the one-time (or on-demand) process of reading
all relevant source files from the Maveric repository, splitting them into
manageable text chunks, generating vector embeddings for each chunk using a
local sentence-transformers model, and storing everything in PostgreSQL with
the pgvector extension.

After ingestion is complete, the database contains a fully searchable vector
store that the retriever (rag_engine/retriever.py) queries at runtime to find
the most relevant documentation context for each user question.

Ingestion Pipeline:
    1. Initialize database tables via SQLAlchemy (if not already created)
    2. Scan only the meaningful folders of the Maveric repo (apps, radp, docs,
       notebooks, srv, tests) — skipping compiled files, caches, and binaries
    3. Filter files by allowed extensions and size limit (50KB max per file)
    4. Read each file and split its text into overlapping chunks of ~1200 chars
       with 200-char overlap to preserve context across chunk boundaries
    5. Display chunk count and confirm before embedding (cost/time awareness)
    6. Embed all chunks locally using 'all-MiniLM-L6-v2' from sentence-transformers
       in batches of 50 for memory efficiency
    7. Store each chunk's text, source path, file type, and embedding vector
       into the document_chunks table in PostgreSQL

Embedding Model:
    Model  : all-MiniLM-L6-v2 (sentence-transformers)
    Dims   : 384
    Size   : ~80MB (downloaded automatically on first run)
    Cost   : Free — runs entirely locally on CPU, no API calls required
    Quality: Sufficient for semantic search over code and documentation

Chunking Strategy:
    chunk_size    = 1200 characters
    chunk_overlap = 200 characters
    Overlapping chunks ensure that important context spanning a chunk boundary
    is not lost. For example, if a function definition starts at character 780
    of a chunk, the next chunk will re-include it from character 650 onwards.

Target Folders:
    Only the following top-level folders are scanned to avoid ingesting
    third-party dependencies, compiled bytecode, or runtime data:
    - apps/       : xApp implementations (energy saving, load balancing, etc.)
    - radp/       : Core RADP library (digital twin, mobility, RF prediction)
    - docs/       : Project documentation (if present)
    - notebooks/  : Jupyter notebooks and test scripts
    - srv/        : Runtime service files (only metadata.json is useful)
    - tests/      : Integration and end-to-end test scripts

Skipped Content:
    - Folders : Venv, venv, __pycache__, .git, node_modules, dist, build,
                cache, .ipynb_checkpoints
    - Files   : Any file larger than 50KB (likely auto-generated or binary)
    - Types   : Any extension not in TARGET_EXTENSIONS (e.g. .pyc, .so, .png)

Re-ingestion:
    Running this script again will delete all existing DocumentChunk records
    and re-ingest from scratch. This is intentional — it ensures the vector
    store stays in sync with the latest state of the Maveric repository.

Usage:
    Run from the project root (maveric_minipilot/):
        python rag_engine/ingest.py

Dependencies:
    - sentence-transformers : local embedding model
    - SQLAlchemy + psycopg2 : PostgreSQL ORM and driver
    - pgvector              : vector storage in PostgreSQL
    - pypdf                 : PDF text extraction
    - python-dotenv         : loads .env config file
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.connection import init_db, SessionLocal
from database.models import DocumentChunk

MAVERIC_REPO_PATH  = os.getenv("MAVERIC_REPO_PATH", "../maveric-repo")
CHUNK_SIZE         = 1200
"""
int: Maximum number of characters per text chunk.
Chosen to balance context richness with embedding quality.
Larger chunks = more context but noisier embeddings.
"""
CHUNK_OVERLAP      = 200
"""
int: Number of characters shared between consecutive chunks.
Prevents loss of context at chunk boundaries. A function definition
or sentence that spans two chunks will appear fully in at least one.
"""
TARGET_FOLDERS = ["apps", "radp", "docs", "notebooks", "srv", "tests", "scripts", "config"]
TARGET_EXTENSIONS = [".md", ".txt", ".py", ".yaml", ".yml", ".json", ".rst", ".ipynb", ".cfg", ".toml", ".ini"]

print("Loading embedding model (first time downloads ~80MB)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
"""
SentenceTransformer: Local embedding model loaded once at module import time.
Reused across all batches to avoid repeated initialization overhead.
Produces 384-dimensional vectors matching the DocumentChunk.embedding column.
"""
print("Embedding model ready.\n")


def get_embedding(text: str) -> list[float]:
    """
    Generate a 384-dimensional embedding vector for a single text string.

    Encodes the input text using the locally loaded 'all-MiniLM-L6-v2'
    sentence-transformers model. The resulting numpy array is converted to
    a plain Python list of floats for compatibility with SQLAlchemy and
    the pgvector column type.

    This function is a thin wrapper around the batch encoding used in the
    main ingest loop. It is provided as a standalone utility for convenience
    (e.g. testing or one-off embedding generation).

    Args:
        text (str):
            The input text to embed. Can be a code snippet, documentation
            paragraph, or any natural language / code string up to ~512 tokens.
            Longer inputs are silently truncated by the model.

    Returns:
        list[float]:
            A list of 384 floating point values representing the semantic
            content of the input text in the model's embedding space.

    Example:
        >>> vec = get_embedding("What is the Digital Twin module?")
        >>> len(vec)
        384
        >>> type(vec[0])
        <class 'float'>
    """
    return embedding_model.encode(text).tolist()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text string into overlapping chunks of a fixed character size.

    Uses a simple sliding window approach: each chunk starts `chunk_size - overlap`
    characters after the previous chunk's start position. This means consecutive
    chunks share `overlap` characters at their boundaries, preserving context
    for content that spans chunk edges.

    This is a character-based splitter (not token or sentence based) chosen for
    its simplicity and determinism. For the Maveric codebase, which consists of
    Python source files and markdown docs, character-based splitting works well
    since meaningful units (functions, paragraphs) are typically well within
    the 800-character window.

    Args:
        text (str):
            The full text content of a source file to be split.
        chunk_size (int, optional):
            Maximum number of characters per chunk. Defaults to CHUNK_SIZE (800).
        overlap (int, optional):
            Number of characters to overlap between consecutive chunks.
            Defaults to CHUNK_OVERLAP (150).

    Returns:
        list[str]:
            A list of text chunks. The last chunk may be shorter than chunk_size
            if the remaining text is less than chunk_size characters. Empty
            chunks (whitespace only) are filtered out by the caller.

    Example:
        >>> chunks = chunk_text("A" * 2000, chunk_size=800, overlap=150)
        >>> len(chunks)
        4
        >>> len(chunks[0])
        800
        >>> chunks[0][-150:] == chunks[1][:150]  # overlap check
        True
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_file(filepath: Path, repo_root: Path) -> list[dict]:
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return []
        chunks = chunk_text(text)
        return [
            {
                "source": str(filepath.relative_to(repo_root)),
                "file_type": filepath.suffix,
                "content": chunk.strip()
            }
            for chunk in chunks if chunk.strip()
        ]
    except Exception as e:
        print(f"  Skipped {filepath.name}: {e}")
        return []


def ingest():
    """
    Main ingestion function — orchestrates the full pipeline from repo scanning
    to vector storage in PostgreSQL.

    This function performs the following steps in sequence:

    1. Database Initialization:
       Calls init_db() which ensures the pgvector extension is enabled and all
       SQLAlchemy model tables (document_chunks, chat_sessions, chat_messages)
       exist in PostgreSQL. Safe to call repeatedly — uses CREATE IF NOT EXISTS.

    2. Repo Validation:
       Resolves MAVERIC_REPO_PATH and exits immediately if the path does not
       exist, providing a clear error message to the user.

    3. File Scanning:
       Iterates through TARGET_FOLDERS in the Maveric repo. For each folder,
       recursively walks all files and applies the following filters:
       - Skips files inside excluded directories (venv, __pycache__, .git, etc.)
       - Skips files larger than 50KB (likely auto-generated or binary content)
       - Skips files whose extension is not in TARGET_EXTENSIONS
       - Handles .pdf files separately using pypdf for text extraction
       Also scans root-level .md and .pdf files (READMEs, Charter, etc.)

    4. Chunk Count and Cost Confirmation:
       Displays the total number of chunks found and prompts the user to confirm
       before proceeding. Since embeddings are generated locally (no API cost),
       this serves as a sanity check before a potentially long operation.

    5. Database Cleanup:
       Deletes all existing DocumentChunk records before inserting new ones.
       This ensures the vector store is always a fresh, consistent reflection
       of the current repo state rather than accumulating stale chunks.

    6. Batch Embedding and Storage:
       Processes chunks in batches of 50. For each batch:
       - Extracts the content strings into a list
       - Calls embedding_model.encode() to generate 384-dim vectors in bulk
         (batch encoding is significantly faster than one-by-one encoding)
       - Creates DocumentChunk ORM objects with content, source, file_type,
         and embedding fields populated
       - Commits the batch to PostgreSQL after each 50-chunk group to avoid
         holding large transactions in memory

    7. Completion:
       Prints a summary of how many chunks were stored and exits cleanly.
       The database is now ready for the retriever to query.

    Raises:
        SystemExit: If the repo path does not exist or no chunks are found.
        Exception:  Any database or embedding error triggers a rollback and
                    re-raises the exception for visibility.

    Side Effects:
        - Drops all existing document_chunks rows before re-ingesting
        - Downloads ~80MB model weights on first run (cached for subsequent runs)
        - Prints progress to stdout throughout execution

    Example:
        Run from project root:
            $ python rag_engine/ingest.py
            Loading embedding model...
            Initialising database...
            Scanning repo: /Users/azanzaman/Desktop/maveric1
              Scanning: apps/
              Loaded: apps/energy_savings/main_app.py (12 chunks)
              ...
            Total chunks to embed: 1065
            Embedding locally — free, no API cost.
            Proceed? (y/n): y
            Cleared 0 existing chunks
              Stored chunks 1-50
              ...
            Done! 1065 chunks stored in PostgreSQL.
    """
    print("Initialising database...")
    init_db()

    repo = Path(MAVERIC_REPO_PATH)
    if not repo.exists():
        print(f"ERROR: Maveric repo not found at {repo.resolve()}")
        sys.exit(1)

    print(f"Scanning repo: {repo.resolve()}\n")

    all_chunks = []

    for folder in TARGET_FOLDERS:
        folder_path = repo / folder
        if not folder_path.exists():
            print(f"  Skipping (not found): {folder}/")
            continue

        print(f"  Scanning: {folder}/")

        for filepath in folder_path.rglob("*"):
            if not filepath.is_file():
                continue

            parts = filepath.parts
            if any(p in ["Venv", "venv", "__pycache__", ".git",
                         "node_modules", "dist", "build", "cache",
                         ".ipynb_checkpoints"]
                   for p in parts):
                continue

            if filepath.stat().st_size > 200_000:
                continue

            if filepath.suffix == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(filepath))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    chunks = chunk_text(text)
                    for chunk in chunks:
                        if chunk.strip():
                            all_chunks.append({
                                "source": str(filepath.relative_to(repo)),
                                "file_type": ".pdf",
                                "content": chunk.strip()
                            })
                    print(f"  Loaded PDF: {filepath.name}")
                except Exception as e:
                    print(f"  Skipped PDF {filepath.name}: {e}")
                continue

            if filepath.suffix not in TARGET_EXTENSIONS:
                continue

            chunks = load_file(filepath, repo)
            if chunks:
                all_chunks.extend(chunks)
                print(f"  Loaded: {chunks[0]['source']} ({len(chunks)} chunks)")

    # Root level md and pdf files
    for filepath in repo.iterdir():
        if not filepath.is_file():
            continue
        if filepath.suffix == ".md":
            chunks = load_file(filepath, repo)
            if chunks:
                all_chunks.extend(chunks)
                print(f"  Loaded root file: {filepath.name}")
        if filepath.suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(filepath))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                chunks = chunk_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        all_chunks.append({
                            "source": filepath.name,
                            "file_type": ".pdf",
                            "content": chunk.strip()
                        })
                print(f"  Loaded root PDF: {filepath.name}")
            except Exception as e:
                print(f"  Skipped root PDF {filepath.name}: {e}")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    if len(all_chunks) == 0:
        print("No chunks found. Check your MAVERIC_REPO_PATH.")
        sys.exit(1)

    print("Embedding locally — free, no API cost.")
    confirm = input("Proceed? (y/n): ")
    if confirm.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    db: Session = SessionLocal()
    try:
        deleted = db.query(DocumentChunk).delete()
        db.commit()
        print(f"Cleared {deleted} existing chunks\n")

        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c["content"] for c in batch]

            # Batch embed locally
            embeddings = embedding_model.encode(texts).tolist()

            for chunk_data, embedding in zip(batch, embeddings):
                db.add(DocumentChunk(
                    source=chunk_data["source"],
                    file_type=chunk_data["file_type"],
                    content=chunk_data["content"],
                    embedding=embedding
                ))

            db.commit()
            print(f"  Stored chunks {i+1}–{min(i+batch_size, len(all_chunks))}")

        print(f"\nDone! {len(all_chunks)} chunks stored in PostgreSQL.")

    except Exception as e:
        db.rollback()
        print(f"ERROR during ingestion: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    ingest()