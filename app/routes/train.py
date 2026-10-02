"""Rotas compatíveis, agora atendidas pelo pipeline persistente."""
import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Form, UploadFile

from app.routes.platform import save_dataset
from app.schemas.dataset import DatasetError
from app.schemas.platform import TrainingConfig
from app.services.dataset_profiler import DatasetProfiler
from app.services.dataset_reader import read_upload
from app.services.legacy_adapter import legacy_analysis, legacy_training
from app.services.model_registry import ArtifactStore, ModelRegistry
from app.services.models import listar_modelos_suportados, normalizar_tipo_modelo, obter_nome_modelo
from app.services.platform_training import train_and_save
from app.services.preprocess import _binarizar_target
from app.services.repository import Repository, get_repository

router = APIRouter(tags=["compatibility"])


@router.post("/analyze")
def analyze(file: UploadFile = File(...)):
    profile = DatasetProfiler().profile(read_upload(file))
    return {"arquivo": file.filename, "modelos_disponiveis": listar_modelos_suportados(), "analise_dataset": legacy_analysis(profile)}


@router.post("/train")
def train(file: UploadFile = File(...), target: str = Form(...), sensitive: str = Form(...),
          model_type: str = Form(""), repository: Repository = Depends(get_repository)):
    record = save_dataset(repository, file)
    frame = ArtifactStore(repository).load("datasets", record.id, "dataset.joblib")
    if target not in frame.columns:
        raise DatasetError("TARGET_NOT_FOUND", "Resultado selecionado não existe na base.")
    try:
        binary, info = _binarizar_target(frame[target])
    except ValueError as exc:
        raise DatasetError("INVALID_TARGET", "Este resultado precisa de uma configuração de classes. Use o fluxo guiado para escolher a classe ou faixa de atenção.") from exc
    config = TrainingConfig(dataset_id=record.id, target=target, positive_class=info["classe_positiva"],
                            sensitive_features=[sensitive] if sensitive else [],
                            identifier_column=record.profile.potential_ids[0] if record.profile.potential_ids else None)
    if model_type:
        config.algorithms = [normalizar_tipo_modelo(model_type)]
    card = train_and_save(repository, config)
    return legacy_training(card, record.profile, {str(k): int(v) for k, v in binary.value_counts().items()})


@router.post("/simulate")
def simulate(data: dict = Body(...), repository: Repository = Depends(get_repository)):
    payload = dict(data)
    model_id = payload.pop("model_id", None)
    model_type = payload.pop("model_type", None)
    if not model_id:
        cards = repository.list("models", 100)
        if model_type:
            cards = [c for c in cards if c["algorithm"] == normalizar_tipo_modelo(model_type)]
        if not cards:
            raise DatasetError("MODEL_NOT_FOUND", "Nenhum modelo salvo está disponível.")
        model_id = cards[0]["id"]
    card, artifact = ModelRegistry(repository).load(model_id)
    probability = float(artifact["pipeline"].predict_proba(pd.DataFrame([payload]))[0, 1])
    return {"modelo": card.algorithm, "modelo_nome": obter_nome_modelo(card.algorithm), "model_id": card.id,
            "probabilidade_evasao": probability}
