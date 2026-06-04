from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.train import router as train_router

app = FastAPI(title="FairDrop API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
