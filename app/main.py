import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.train import router as train_router


def _get_allowed_origins():
    default_origins = {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    }
    extra_origins = {
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
        if origin.strip()
    }
    return sorted(default_origins | extra_origins)


app = FastAPI(title="FairDrop API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
