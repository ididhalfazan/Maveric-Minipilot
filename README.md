# Maveric-Minipilot
Maveric MiniPilot is an AI-powered interactive chatbot designed to guide users through the Maveric (RIC Algorithm Development Platform) workflow end-to-end.

# Maveric MiniPilot

AI-powered chatbot for the Maveric RIC Algorithm Development Platform.

## Stack
- FastAPI — backend REST API
- PostgreSQL + pgvector — vector store + chat history
- sentence-transformers (all-MiniLM-L6-v2) — local embeddings
- Groq (llama3-70b-8192) — LLM
- Streamlit — frontend UI

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure
Copy `.env.example` to `.env` and fill in your keys:
```
OPENAI_API_KEY=
GROQ_API_KEY=
MAVERIC_REPO_PATH=/path/to/maveric/repo
DATABASE_URL=postgresql://maveric_user:maveric_pass@localhost:5432/maveric_minipilot
```

## Run
```bash
# One-time ingestion
python rag_engine/ingest.py

# Start backend
uvicorn backend.main:app --reload --port 8000

# Start frontend
streamlit run frontend/app.py
```
EOF
