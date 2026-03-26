# Maveric MiniPilot — Full Project Documentation

> AI-powered interactive chatbot for the Maveric RIC Algorithm Development Platform.  



## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Database Schema](#5-database-schema)
6. [RAG Pipeline](#6-rag-pipeline)
7. [API Endpoints](#7-api-endpoints)
8. [Frontend](#8-frontend)
9. [LLM Integration](#9-llm-integration)
10. [Setup & Installation](#10-setup--installation)
11. [Running the Project](#11-running-the-project)
12. [Verification & Debugging](#12-verification--debugging)
13. [Environment Variables](#13-environment-variables)
14. [Known Limitations](#14-known-limitations)
15. [Future Improvements](#15-future-improvements)

---

## 1. Project Overview

Maveric MiniPilot is an AI-powered chatbot that serves as an intelligent guide for the Maveric RIC Algorithm Development Platform. Instead of reading through source code and documentation manually, users can ask natural language questions and receive grounded, accurate answers derived directly from the Maveric repository.

### What it does

- Answers questions about the full Maveric workflow — simulation setup, UE tracks generation, Digital Twin training, RF Prediction, and job orchestration
- Provides module-specific deep dives into Digital Twin, RF Prediction, UE Tracks Generation, and Orchestration
- Maintains conversation context across multiple turns within a session
- Shows the exact source files used to generate each answer
- Runs entirely for free — local embeddings via sentence-transformers, LLM via Groq free tier

### What it does NOT do

- It does not run the Maveric simulation pipeline
- It does not connect to any Maveric API or microservice at runtime
- It does not modify the Maveric repository in any way
- The Maveric repo is treated as a read-only documentation source

---

## 2. Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit Frontend            │
│           frontend/app.py               │
│   chat UI · module selector · sources   │
└──────────────┬──────────────────────────┘
               │  HTTP POST /chat
               ▼
┌─────────────────────────────────────────┐
│           FastAPI Backend               │
│           backend/main.py               │
│  /chat · /clear · /health · /sessions   │
└──────┬───────────────────┬─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐   ┌─────────────────────┐
│  RAG Engine │   │     PostgreSQL       │
│             │   │                     │
│ retriever   │◄──│  document_chunks    │
│ ingest      │   │  chat_sessions      │
└──────┬──────┘   │  chat_messages      │
       │          └─────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│     all-MiniLM-L6-v2 (local)           │
│     sentence-transformers               │
│     384-dim embeddings · free · fast    │
└─────────────────────────────────────────┘
       │
       ▼ (assembled prompt)
┌─────────────────────────────────────────┐
│     Groq API — llama3-70b-8192          │
│     generates grounded answer           │
└─────────────────────────────────────────┘
       │
       ▼ (answer + sources)
┌─────────────────────────────────────────┐
│     Maveric Repo (read-only source)     │
│  apps/ · radp/ · docs/ · tests/        │
│  ingested once via ingest.py            │
└─────────────────────────────────────────┘
```

### How a single request flows

1. User types a question in Streamlit
2. Streamlit POSTs to FastAPI `/chat`
3. FastAPI calls `retrieve()` in the RAG engine
4. RAG engine embeds the question using `all-MiniLM-L6-v2` locally
5. pgvector finds the 4 most semantically similar chunks in PostgreSQL
6. FastAPI loads conversation history from `chat_messages` table
7. FastAPI assembles: system prompt + retrieved chunks + history + question
8. Assembled prompt is sent to Groq LLaMA3
9. Answer is returned, saved to PostgreSQL, sent back to Streamlit
10. Streamlit renders the answer with a collapsible sources panel

---

## 3. Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat UI, module selector, source display |
| Backend | FastAPI + uvicorn | REST API, request orchestration |
| Database | PostgreSQL 17 | Persistent storage for embeddings and chat |
| Vector search | pgvector | Cosine similarity search on embeddings |
| ORM | SQLAlchemy | Python-to-PostgreSQL interface |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local 384-dim text embeddings |
| LLM | Groq `llama3-70b-8192` | Answer generation |
| PDF parsing | pypdf | Extract text from PDF files in repo |
| Config | python-dotenv | Environment variable management |

---

## 4. Project Structure

```
maveric_minipilot/
│
├── .env                        # API keys and config (never commit this)
├── .env.example                # Template showing required variables
├── .gitignore                  # Excludes venv, .env, __pycache__
├── requirements.txt            # All Python dependencies
├── README.md                   # Quick start guide
│
├── database/
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM models (3 tables)
│   └── connection.py           # DB engine, SessionLocal, init_db(), get_db()
│
├── rag_engine/
│   ├── __init__.py
│   ├── ingest.py               # One-time repo ingestion pipeline
│   └── retriever.py            # Query-time cosine similarity retrieval
│
├── backend/
│   ├── __init__.py
│   └── main.py                 # FastAPI app with all endpoints
│
├── frontend/
│   └── app.py                  # Streamlit chat UI
│
├── Dockerfile.backend          # Docker image for backend
├── Dockerfile.frontend         # Docker image for frontend
└── docker-compose.yml          # Orchestrates all services
```

---

## 5. Database Schema

### `document_chunks`
Stores chunked content from the Maveric repo with vector embeddings.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment primary key |
| `source` | VARCHAR(500) | Relative file path e.g. `radp/digital_twin/rf/bayesian/bayesian_engine.py` |
| `file_type` | VARCHAR(20) | File extension e.g. `.py`, `.md`, `.yaml` |
| `content` | TEXT | Raw text of the chunk (~1200 chars) |
| `embedding` | VECTOR(384) | Embedding from `all-MiniLM-L6-v2` |
| `created_at` | TIMESTAMP | Insertion time |

### `chat_sessions`
Tracks individual user conversations.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment primary key |
| `session_id` | VARCHAR(100) UNIQUE | UUID generated by Streamlit frontend |
| `created_at` | TIMESTAMP | Session creation time |

### `chat_messages`
Stores every message in every session.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment primary key |
| `session_id` | VARCHAR(100) FK | References `chat_sessions.session_id` |
| `role` | VARCHAR(20) | Either `user` or `assistant` |
| `content` | TEXT | Full message text |
| `created_at` | TIMESTAMP | Message timestamp |

> **Cascade delete**: deleting a `ChatSession` automatically deletes all its `ChatMessage` records.

---

## 6. RAG Pipeline

RAG stands for Retrieval Augmented Generation. Instead of relying on the LLM's training knowledge, we inject actual Maveric documentation into every prompt.

### Phase 1 — Ingestion (run once)

```
Maveric repo on disk
        ↓
rag_engine/ingest.py scans these folders:
  apps/ · radp/ · docs/ · notebooks/ · srv/ · tests/

Filters applied:
  ✓ extensions: .py .md .yaml .yml .json .rst .txt .pdf
  ✗ skip: venv/ __pycache__ .git node_modules
  ✗ skip: files over 50KB

Each file is:
  1. Read as UTF-8 text
  2. Split into 1200-char chunks with 200-char overlap
  3. Embedded by all-MiniLM-L6-v2 → 384 floats
  4. Stored in document_chunks table

Result: 1,065 chunks from 102 source files
Cost: $0.00 (fully local)
Time: ~2 minutes
```

### Phase 2 — Retrieval (every request)

```
User question: "How does the Bayesian engine work?"
        ↓
get_embedding(question) → [0.023, -0.117, 0.841, ...] (384 floats)
        ↓
PostgreSQL query:
  SELECT source, content,
    1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity
  FROM document_chunks
  ORDER BY embedding <=> CAST(:query_vector AS vector)
  LIMIT 4
        ↓
Top 4 chunks returned:
  score=0.91  source=radp/digital_twin/rf/bayesian/bayesian_engine.py
  score=0.84  source=radp/example_bayesian_engine_driver_script.py
  score=0.80  source=radp/digital_twin/rf/bayesian/tests/test_bayesian_engine.py
  score=0.74  source=radp/digital_twin/__init__.py
```

### Phase 3 — Generation

```
Prompt assembled:
  [system]    SYSTEM_PROMPT (role + instructions)
  [system]    Retrieved chunk 1 with source label
              Retrieved chunk 2 with source label
              Retrieved chunk 3 with source label
              Retrieved chunk 4 with source label
  [user]      message turn 1 (from history)
  [assistant] answer turn 1 (from history)
  ...up to 10 turns of history...
  [user]      "How does the Bayesian engine work?"
        ↓
Groq llama3-70b-8192 → grounded answer citing actual Maveric code
```

### Retrieval type

This implementation uses **dense vector retrieval** (semantic search), not BM25 (keyword search) or live/agentic RAG. The index is built once and queried at runtime. Semantic search understands meaning — a question about "signal propagation" correctly matches chunks about "RF prediction" even without exact keyword overlap.

---

## 7. API Endpoints

All endpoints are defined in `backend/main.py` and served on `http://localhost:8000`.

### `POST /chat`

Main endpoint. Receives a user message, runs RAG retrieval, calls Groq, returns answer.

**Request body:**
```json
{
  "message": "How does the Bayesian engine work?",
  "session_id": "3f2a1b4c-8d9e-4f5a-b6c7",
  "module_focus": "digital_twin"
}
```

**Response:**
```json
{
  "answer": "The Bayesian engine in Maveric is implemented in...",
  "session_id": "3f2a1b4c-8d9e-4f5a-b6c7",
  "sources": [
    "radp/digital_twin/rf/bayesian/bayesian_engine.py",
    "radp/example_bayesian_engine_driver_script.py"
  ]
}
```

**`module_focus` options:**

| Value | Effect |
|-------|--------|
| `digital_twin` | Prepends "Digital Twin module:" to query |
| `rf_prediction` | Prepends "RF Prediction module:" to query |
| `ue_tracks` | Prepends "UE Tracks Generation module:" to query |
| `orchestration` | Prepends "Orchestration module:" to query |
| `null` | No prefix, general query |

---

### `POST /clear`

Deletes a session and all its messages from the database.

**Request body:**
```json
{ "session_id": "3f2a1b4c-8d9e-4f5a-b6c7" }
```

**Response:**
```json
{ "cleared": "3f2a1b4c-8d9e-4f5a-b6c7" }
```

---

### `GET /health`

Health check used by Streamlit on every page load.

**Response:**
```json
{ "status": "ok", "sessions_in_db": 3 }
```

---

### `GET /sessions`

Lists all active session IDs in the database.

**Response:**
```json
{ "sessions": ["3f2a1b4c-...", "9a8b7c6d-..."] }
```

---

## 8. Frontend

`frontend/app.py` is a Streamlit application that provides the chat interface.

### Key features

- **Chat window** — renders conversation history with role-based message bubbles
- **Module selector** — radio buttons in sidebar to focus on a specific Maveric module
- **Quick questions** — preset buttons for common Maveric queries
- **Sources expander** — collapsible panel under each answer showing which repo files were used
- **Clear chat** — deletes session from DB and resets UI state
- **Health check** — automatically detects if backend is down and shows an error

### State management

Streamlit uses `st.session_state` for in-memory UI state:

| Key | Type | Purpose |
|-----|------|---------|
| `session_id` | str (UUID4) | Unique ID for this browser session |
| `messages` | list | Local copy of conversation for rendering |
| `module_focus` | str or None | Currently selected module |
| `pending_message` | str | Quick question waiting to be sent |

### Backend connection

```python
BACKEND_URL = "http://localhost:8000"

# Health check on every page load
requests.get(f"{BACKEND_URL}/health", timeout=2)

# Send message
requests.post(f"{BACKEND_URL}/chat", json={
    "message": user_input,
    "session_id": st.session_state.session_id,
    "module_focus": st.session_state.module_focus
}, timeout=30)
```

---

## 9. LLM Integration

### Current setup — Groq

```python
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    temperature=0.3,   # low = more factual, less creative
    max_tokens=800
)
```

`temperature=0.3` is intentionally low to keep answers grounded in the retrieved documentation rather than hallucinating.

### Planned — OpenAI with local fallback

The spec requires `OpenAI API with local fallback support`. The fallback chain:

```python
def call_llm(messages):
    # 1. Try OpenAI (primary)
    try:
        return openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, ...)
    except:
        pass

    # 2. Try Groq (cloud fallback)
    try:
        return groq_client.chat.completions.create(
            model="llama3-70b-8192", messages=messages, ...)
    except:
        pass

    # 3. Try Ollama (true local fallback)
    return requests.post("http://localhost:11434/api/chat",
        json={"model": "llama3", "messages": messages, "stream": False})
```

Groq is used as primary in the current build due to OpenAI billing constraints during development.

---

## 10. Setup & Installation

### Prerequisites

- Mac with Homebrew installed
- Python 3.11 (via pyenv)
- PostgreSQL 17 (via Homebrew)
- Groq API key (free at console.groq.com)

### Step 1 — Clone the repo

```bash
git clone https://github.com/ididhalfazan/maveric-minipilot.git
cd maveric-minipilot
```

### Step 2 — Python environment

```bash
pyenv local 3.11.9
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3 — PostgreSQL setup

```bash
brew services start postgresql@17
psql postgres
```

Inside psql:
```sql
CREATE DATABASE maveric_minipilot;
CREATE USER maveric_user WITH PASSWORD 'maveric_pass';
GRANT ALL PRIVILEGES ON DATABASE maveric_minipilot TO maveric_user;
\c maveric_minipilot
GRANT ALL ON SCHEMA public TO maveric_user;
CREATE EXTENSION vector;
\q
```

### Step 4 — Environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_your_key_here
MAVERIC_REPO_PATH=/absolute/path/to/your/maveric/repo
DATABASE_URL=postgresql://maveric_user:maveric_pass@localhost:5432/maveric_minipilot
```

### Step 5 — Ingest the Maveric repo

```bash
python rag_engine/ingest.py
```

This runs once. Downloads the embedding model (~80MB on first run), scans the Maveric repo, and stores 1,065 chunks in PostgreSQL. Takes about 2 minutes.

---

## 11. Running the Project

Open two terminal windows, both with `venv` activated and in the project root.

**Terminal 1 — Backend:**
```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
source venv/bin/activate
streamlit run frontend/app.py
```

Open your browser at **http://localhost:8501**

The backend API docs (Swagger UI) are available at **http://localhost:8000/docs**

---

## 12. Verification & Debugging

### Check database

```bash
psql postgres
\c maveric_minipilot

SELECT COUNT(*) FROM document_chunks;      -- should be 1065
SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL;  -- should be 0
SELECT source, LEFT(content, 80) FROM document_chunks LIMIT 5;
\q
```

### Test retriever directly

```python
# debug/test_retriever.py
import sys
sys.path.append(".")
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(".env"))

from database.connection import SessionLocal
from rag_engine.retriever import retrieve

db = SessionLocal()
chunks = retrieve("How does the Bayesian engine work?", k=4, db=db)
for c in chunks:
    print(f"score={c['score']:.3f}  source={c['source']}")
db.close()
```

```bash
python debug/test_retriever.py
```

### Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the Digital Twin?", "session_id": "test-1"}' \
  | python3 -m json.tool
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## 13. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key from console.groq.com |
| `MAVERIC_REPO_PATH` | Yes | Absolute path to Maveric repo root |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `OPENAI_API_KEY` | Optional | For OpenAI fallback (future) |

---

## 14. Known Limitations

- **Conversation history resets on backend restart** — history is stored in PostgreSQL so it survives, but the Streamlit UI state (the rendered messages) resets on page refresh
- **No MCP server implemented** — the spec mentioned MCP integration but it was explicitly marked as "not needed for now"
- **No formal evaluation metrics** — retrieval quality is verified manually via similarity scores; no RAGAS or automated evaluation pipeline
- **Groq as primary LLM** — spec requires OpenAI with local fallback; Groq is used as primary due to billing constraints during development
- **Static RAG index** — the index must be manually re-ingested when the Maveric repo is updated (`python rag_engine/ingest.py`)
- **In-memory UI state** — Streamlit loses rendered messages on page refresh (the DB records are intact, the display is not)

---

## 15. Future Improvements

| Improvement | Description |
|-------------|-------------|
| OpenAI + local fallback | Implement the full `call_llm()` fallback chain as designed |
| MCP server | Expose Maveric simulation tools as MCP-callable functions |
| RAGAS evaluation | Automated retrieval quality metrics — faithfulness, relevance, precision |
| Re-ingestion on change | Watch Maveric repo for changes and auto re-ingest modified files |
| Hybrid search | Combine dense vector search with BM25 for better keyword recall |
| Auth | Add user authentication so sessions are tied to individual users |
| Docker deploy | Use `docker-compose up` for one-command deployment |
| Streaming responses | Stream LLM tokens to Streamlit in real time instead of waiting for full response |

---

## Quick Reference

```bash
# Activate environment
source venv/bin/activate

# Re-ingest repo (after Maveric repo updates)
python rag_engine/ingest.py

# Start backend
uvicorn backend.main:app --reload --port 8000

# Start frontend
streamlit run frontend/app.py

# Push to GitHub
git add . && git commit -m "your message" && git push
```




