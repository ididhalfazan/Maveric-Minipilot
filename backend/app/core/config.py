# backend/app/core/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parents[3] / ".env")

class Settings:
    GROQ_API_KEY: str       = os.getenv("GROQ_API_KEY")
    DATABASE_URL: str       = os.getenv("DATABASE_URL")
    MAVERIC_REPO_PATH: str  = os.getenv("MAVERIC_REPO_PATH")
    GROQ_MODEL: str         = "llama-3.3-70b-versatile"
    SIMILARITY_THRESHOLD: float = 0.5
    RETRIEVAL_K: int        = 4

    MAVERIC_MODULES: dict = {
        "digital_twin":  "Digital Twin module",
        "rf_prediction": "RF Prediction module",
        "ue_tracks":     "UE Tracks Generation module",
        "orchestration": "Orchestration / Job Orchestration module",
        "workflow":      "full Maveric end-to-end workflow",
    }

    SYSTEM_PROMPT: str = """You are Maveric MiniPilot, an AI assistant that guides users through
the Maveric RIC Algorithm Development Platform.

You help users understand:
- Full workflow: simulation setup, UE tracks generation, Digital Twin training,
  RF Prediction, and job orchestration
- Module deep-dives: Digital Twin, RF Prediction, UE Tracks Generation, Orchestration
- Inputs, outputs, internal flows, and microservice references for each module

Always base your answers on the provided documentation context.
If the context does not contain enough information, say so honestly.
Be concise, clear, and guide the user step by step."""

settings = Settings()