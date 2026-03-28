"""
One-time ingestion pipeline — reads the Maveric repo, chunks files,
embeds them locally, and stores everything in PostgreSQL.

Run from project root:
    python -m backend.app.rag.ingestion
"""
import sys
from pathlib import Path
from sqlalchemy.orm import Session
from backend.app.core.config import settings
from backend.app.core.database import init_db, SessionLocal
from backend.app.models.conversation import DocumentChunk
from backend.app.rag.embeddings import embedding_model

TARGET_FOLDERS    = ["apps", "radp", "docs", "notebooks", "srv", "tests"]
TARGET_EXTENSIONS = [".md", ".txt", ".py", ".yaml", ".yml", ".json", ".rst"]
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 150


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of CHUNK_SIZE characters."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_file(filepath: Path, repo_root: Path) -> list[dict]:
    """Read a file and return its chunks with metadata."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return []
        return [
            {
                "source": str(filepath.relative_to(repo_root)),
                "file_type": filepath.suffix,
                "content": chunk.strip()
            }
            for chunk in chunk_text(text) if chunk.strip()
        ]
    except Exception as e:
        print(f"  Skipped {filepath.name}: {e}")
        return []


def ingest():
    """
    Full ingestion pipeline:
        1. Init DB tables
        2. Scan Maveric repo folders
        3. Chunk + embed all files locally
        4. Store in document_chunks table
    """
    print("Initialising database...")
    init_db()

    repo = Path(settings.MAVERIC_REPO_PATH)
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
            if filepath.stat().st_size > 50_000:
                continue
            if filepath.suffix == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(filepath))
                    text = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                    for chunk in chunk_text(text):
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
                print(f"  Loaded: {chunks[0]['source']} "
                      f"({len(chunks)} chunks)")

    # Root level md and pdf files
    for filepath in repo.iterdir():
        if not filepath.is_file():
            continue
        if filepath.suffix == ".md":
            chunks = load_file(filepath, repo)
            if chunks:
                all_chunks.extend(chunks)
                print(f"  Loaded root: {filepath.name}")
        if filepath.suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(filepath))
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                for chunk in chunk_text(text):
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

    if not all_chunks:
        print("No chunks found. Check MAVERIC_REPO_PATH in .env")
        sys.exit(1)

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
            embeddings = embedding_model.encode(
                [c["content"] for c in batch]
            ).tolist()

            for chunk_data, embedding in zip(batch, embeddings):
                db.add(DocumentChunk(
                    source=chunk_data["source"],
                    file_type=chunk_data["file_type"],
                    content=chunk_data["content"],
                    embedding=embedding
                ))
            db.commit()
            print(f"  Stored chunks "
                  f"{i+1}–{min(i+batch_size, len(all_chunks))}")

        print(f"\nDone! {len(all_chunks)} chunks stored in PostgreSQL.")

    except Exception as e:
        db.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    ingest()