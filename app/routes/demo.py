from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.schemas.dataset import DatasetError
from app.schemas.platform import PredictionBatch
from app.services.dataset_reader import read_dataset
from app.services.demo import ASSET_DIR, DEMO_DATASET_ID
from app.services.prediction import RiskPredictor
from app.services.repository import Repository, get_repository

router = APIRouter(tags=["demo"])


@router.get("/demo")
def demo(repository: Repository = Depends(get_repository)):
    models = [m for m in repository.list("models", 10000) if m.get("is_demo")]
    return {"dataset_id": DEMO_DATASET_ID, "models": models,
            "message": "Exemplos com dados fictícios. Experimente antes de enviar sua própria base."}


@router.post("/demo/predict", response_model=PredictionBatch)
def predict_demo(model_id: str | None = None, repository: Repository = Depends(get_repository)):
    models = [m for m in repository.list("models", 10000) if m.get("is_demo")]
    if model_id:
        models = [model for model in models if model["id"] == model_id]
    if not models:
        raise DatasetError("DEMO_UNAVAILABLE", "Os modelos de exemplo foram excluídos. Use uma base histórica para treinar um novo modelo.")
    frame = read_dataset((ASSET_DIR / "current_students.csv").read_bytes(), "current_students.csv")
    return RiskPredictor(repository).predict(models[0]["id"], frame, "Demonstração · 30 alunos fictícios")


@router.get("/demo/files/{kind}")
def demo_file(kind: str):
    names = {"historical": "historical_students.csv", "current": "current_students.csv"}
    if kind not in names:
        raise DatasetError("NOT_FOUND", "Exemplo não encontrado.", 404)
    return FileResponse(ASSET_DIR / names[kind], media_type="text/csv", filename=names[kind])
