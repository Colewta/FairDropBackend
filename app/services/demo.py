"""Exemplos fictícios e modelos distribuídos com a aplicação, sem dados pessoais."""
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from app.schemas.platform import ModelCard
from app.services.dataset_profiler import DatasetProfiler
from app.services.dataset_reader import read_dataset
from app.services.model_registry import ArtifactStore
from app.services.repository import Repository

ASSET_DIR = Path(__file__).resolve().parents[2] / "demo_assets"
DEMO_DATASET_ID = uuid5(NAMESPACE_URL, "fairdrop/demo/historical/v2").hex


def install_demo(repository: Repository) -> None:
    if not (ASSET_DIR / "historical_students.csv").exists():
        return
    marker = ArtifactStore(repository).root / "demo-installed-v2"
    if marker.exists():
        return
    frame = read_dataset((ASSET_DIR / "historical_students.csv").read_bytes(), "historical_students.csv")
    record = {"id": DEMO_DATASET_ID, "name": "Exemplo fictício · histórico de alunos", "created_at": "2026-01-01T00:00:00+00:00",
              "is_demo": True, "profile": DatasetProfiler().profile(frame, "situacao").model_dump(mode="json")}
    store = ArtifactStore(repository)
    store.save("datasets", DEMO_DATASET_ID, "dataset.joblib", frame)
    repository.put("datasets", record)
    for path in sorted(ASSET_DIR.glob("*/metadata.json")):
        import joblib
        card = ModelCard.model_validate_json(path.read_text(encoding="utf-8"))
        card.institution_id = repository.institution_id
        bundle = joblib.load(path.parent / "model.joblib")
        store.save("models", card.id, "model.joblib", bundle)
        store.save("models", card.id, "metadata.json", card.model_dump(mode="json"))
        repository.put("models", card.model_dump(mode="json"))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("Modelos demonstrativos instalados; não recriar automaticamente após exclusão.", encoding="utf-8")
