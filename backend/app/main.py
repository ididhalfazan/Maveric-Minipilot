# backend/app/main.py
print(">>> Running NEW modular backend <<<")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.v1.router import router
from backend.app.core.database import init_db

app = FastAPI(title="Maveric MiniPilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app",
                host="0.0.0.0", port=8000, reload=True)
