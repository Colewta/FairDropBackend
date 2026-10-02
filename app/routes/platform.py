from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile

from app.schemas.platform import (DatasetRecord, ModelCard, PredictionBatch, PredictionPage,
                                  StudentDetail, TrainingConfig, TrainingRun, MitigationRequest, ScenarioRequest, TutorReview)
from app.schemas.dataset import DatasetError
from app.services.dataset_profiler import DatasetProfiler
from app.services.dataset_reader import read_upload
from app.services.model_registry import ArtifactStore, ModelRegistry
from app.services.platform_training import execute_training_run, train_and_save
from app.services.prediction import RiskPredictor
from app.services.repository import Repository, get_repository, new_id, now
import pandas as pd
from app.services.numeric import parse_numeric

router = APIRouter(tags=["platform"])


def save_dataset(repository: Repository, file: UploadFile) -> DatasetRecord:
    frame = read_upload(file)
    record = DatasetRecord(id=new_id(), name=file.filename or "Base", created_at=now(), profile=DatasetProfiler().profile(frame))
    ArtifactStore(repository).save("datasets", record.id, "dataset.joblib", frame)
    repository.put("datasets", record.model_dump(mode="json"))
    return record


@router.post("/datasets", response_model=DatasetRecord, status_code=201)
def upload_dataset(file: UploadFile = File(...), repository: Repository = Depends(get_repository)):
    return save_dataset(repository, file)


@router.get("/datasets", response_model=list[DatasetRecord])
def list_datasets(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0), repository: Repository = Depends(get_repository)):
    return repository.list("datasets", limit, offset)


@router.get("/datasets/{identifier}", response_model=DatasetRecord)
def get_dataset(identifier: str, target: str | None = None, repository: Repository = Depends(get_repository)):
    record = repository.get("datasets", identifier)
    if target is not None:
        frame = ArtifactStore(repository).load("datasets", record["id"], "dataset.joblib")
        record["profile"] = DatasetProfiler().profile(frame, target).model_dump(mode="json")
    return record


@router.get("/datasets/{identifier}/values")
def column_values(identifier: str, column: str, repository: Repository = Depends(get_repository)):
    record = repository.get("datasets", identifier)
    frame = ArtifactStore(repository).load("datasets", record["id"], "dataset.joblib")
    from app.schemas.dataset import DatasetError
    if column not in frame.columns or column in record["profile"]["potential_ids"]:
        raise DatasetError("INVALID_COLUMN", "Selecione uma coluna de resultado ou grupos, sem identificadores.")
    values = frame[column].dropna().astype(str).value_counts()
    if len(values) > 30:
        numeric = parse_numeric(frame[column]).dropna()
        if len(numeric) == frame[column].notna().sum():
            return {"values": [], "numeric": {"min": float(numeric.min()), "max": float(numeric.max()), "suggested_cut": float(numeric.median())},
                    "message": "Esta coluna contém números. Escolha uma faixa de atenção para transformar sua pergunta em duas situações."}
        raise DatasetError("TOO_MANY_CLASSES", "Mais de 30 valores distintos; defina categorias antes de treinar.")
    return {"values": [{"value": str(value), "count": int(count)} for value, count in values.items()]}


@router.get("/datasets/{identifier}/preview")
def dataset_preview(identifier: str, repository: Repository = Depends(get_repository)):
    record = repository.get("datasets", identifier)
    frame = ArtifactStore(repository).load("datasets", record["id"], "dataset.joblib")
    masked = record["profile"]["potential_ids"]
    rows = [{column: "••••" if column in masked else None if pd.isna(value) else str(value)
             for column, value in row.items()} for row in frame.head(5).to_dict(orient="records")]
    return {"columns": list(frame.columns), "rows": rows, "masked_columns": masked}


@router.delete("/datasets/{identifier}", status_code=204)
def delete_dataset(identifier: str, repository: Repository = Depends(get_repository)):
    record = repository.get("datasets", identifier)
    ArtifactStore(repository).path("datasets", record["id"], "dataset.joblib").unlink(missing_ok=True)
    repository.remove("datasets", identifier)


@router.post("/train-automl", response_model=ModelCard)
def train_automl(config: TrainingConfig, repository: Repository = Depends(get_repository)):
    return train_and_save(repository, config)


@router.post("/training-runs", response_model=TrainingRun, status_code=202)
def start_training(config: TrainingConfig, tasks: BackgroundTasks, repository: Repository = Depends(get_repository)):
    repository.get("datasets", config.dataset_id)
    run = TrainingRun(id=new_id(), status="queued", created_at=now())
    repository.put("training_runs", {**run.model_dump(), "config": config.model_dump(mode="json")})
    tasks.add_task(execute_training_run, repository, run.id, config)
    return run


@router.get("/training-runs/{identifier}", response_model=TrainingRun)
def get_training(identifier: str, repository: Repository = Depends(get_repository)):
    return repository.get("training_runs", identifier)


@router.get("/models", response_model=list[ModelCard])
def list_models(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0), repository: Repository = Depends(get_repository)):
    return repository.list("models", limit, offset)


@router.get("/models/{identifier}", response_model=ModelCard)
def get_model(identifier: str, repository: Repository = Depends(get_repository)):
    return repository.get("models", identifier)


@router.delete("/models/{identifier}", status_code=204)
def delete_model(identifier: str, repository: Repository = Depends(get_repository)):
    ModelRegistry(repository).delete(identifier)


@router.post("/predict", response_model=PredictionBatch, status_code=201)
def predict(file: UploadFile = File(...), model_id: str = Form(...), repository: Repository = Depends(get_repository)):
    return RiskPredictor(repository).predict(model_id, read_upload(file), file.filename or "Alunos atuais")


@router.get("/predictions", response_model=list[PredictionBatch])
def list_batches(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0), repository: Repository = Depends(get_repository)):
    return repository.list("prediction_batches", limit, offset)


@router.get("/predictions/{identifier}", response_model=PredictionBatch)
def get_batch(identifier: str, repository: Repository = Depends(get_repository)):
    return repository.get("prediction_batches", identifier)


@router.get("/predictions/{identifier}/students", response_model=PredictionPage)
def list_students(identifier: str, page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
                  risk: str | None = Query(None, pattern="^(low|medium|high)$"), q: str = Query("", max_length=200),
                  group_column: str | None = Query(None, max_length=200), group_value: str | None = Query(None, max_length=500),
                  repository: Repository = Depends(get_repository)):
    total, items = repository.prediction_page(identifier, page, page_size, risk, q, group_column, group_value)
    return PredictionPage(total=total, items=items, page=page, page_size=page_size)


@router.delete("/predictions/{identifier}", status_code=204)
def delete_batch(identifier: str, repository: Repository = Depends(get_repository)):
    repository.delete_batch(identifier)


@router.get("/students/{identifier}", response_model=StudentDetail)
def get_student(identifier: str, repository: Repository = Depends(get_repository)):
    return RiskPredictor(repository).explain_student(identifier)


@router.post("/models/{identifier}/mitigate", response_model=TrainingRun, status_code=202)
def mitigate_model(identifier: str, request: MitigationRequest, tasks: BackgroundTasks, repository: Repository = Depends(get_repository)):
    card = ModelCard.model_validate(repository.get("models", identifier))
    if request.feature not in card.sensitive_features:
        raise DatasetError("INVALID_MITIGATION_GROUP", "Escolha um atributo confirmado na avaliação de equidade deste modelo.")
    repository.get("datasets", card.dataset_id)
    config = card.config.model_copy(update={"mitigation": "reweighing", "mitigation_features": [request.feature],
        "strategy": request.strategy, "name": f"Mitigação · {card.name}"[:160]})
    return start_training(config, tasks, repository)


@router.post("/students/{identifier}/scenario")
def simulate_student(identifier: str, request: ScenarioRequest, repository: Repository = Depends(get_repository)):
    return RiskPredictor(repository).scenario(identifier, request.feature, request.value)


@router.get("/students/{identifier}/review", response_model=TutorReview)
def read_review(identifier: str, repository: Repository = Depends(get_repository)):
    row = repository.student(identifier)
    try:
        return repository.get("tutor_reviews", row["id"])
    except DatasetError as exc:
        if exc.code != "NOT_FOUND":
            raise
        return TutorReview()


@router.put("/students/{identifier}/review", response_model=TutorReview)
def save_review(identifier: str, review: TutorReview, repository: Repository = Depends(get_repository)):
    row = repository.student(identifier)
    review.updated_at = now()
    repository.put("tutor_reviews", {"id": row["id"], **review.model_dump(mode="json")})
    return review


@router.get("/dashboard")
def dashboard(repository: Repository = Depends(get_repository)):
    batches = repository.list("prediction_batches", 1)
    return {"latest_batch": batches[0] if batches else None,
            "message": "Resumo do lote mais recente; cada envio representa uma fotografia dos alunos."}
