import os
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.train import router as train_router
from app.routes.datasets import router as datasets_router
from app.routes.platform import router as platform_router
from app.core.config import API_TOKEN
from app.schemas.dataset import DatasetError
from app.services.repository import get_repository
from app.services.demo import install_demo
from app.routes.demo import router as demo_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = get_repository()
    if os.getenv("FAIRDROP_DEMO_ENABLED", "true").lower() == "true":
        install_demo(repository)
    for run in repository.list("training_runs", 10000):
        if run["status"] in {"queued", "running"}:
            run.update(status="failed", error="Treinamento interrompido pelo reinício do serviço. Inicie uma nova execução.")
            repository.put("training_runs", run)
    yield


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


app = FastAPI(title="FairDrop API", version="2.1.0", lifespan=lifespan)


@app.middleware("http")
async def protect_api(request: Request, call_next):
    if API_TOKEN and request.method != "OPTIONS" and request.url.path not in {"/health", "/docs", "/openapi.json", "/redoc"}:
        supplied = request.headers.get("Authorization", "")
        if not secrets.compare_digest(supplied, f"Bearer {API_TOKEN}"):
            return JSONResponse(status_code=401, content={"detail": {"error": "UNAUTHORIZED", "message": "Informe a chave de acesso da instituição."}})
    return await call_next(request)


@app.exception_handler(DatasetError)
async def domain_error(request: Request, exc: DatasetError):
    return JSONResponse(status_code=exc.status_code, content={"detail": {"error": exc.code, "message": str(exc)}})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logger.error("Falha na API: type=%s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": {"error": "INTERNAL_ERROR", "message": "Não foi possível concluir a operação."}})

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router)
app.include_router(datasets_router)
app.include_router(platform_router)
app.include_router(demo_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
