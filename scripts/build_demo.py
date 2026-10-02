"""Regera artefatos demonstrativos com versões do requirements.txt. Dados 100% sintéticos."""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas.platform import TrainingConfig
from app.services.dataset_profiler import DatasetProfiler
from app.services.demo import ASSET_DIR, DEMO_DATASET_ID
from app.services.model_registry import ArtifactStore
from app.services.platform_training import train_and_save
from app.services.repository import Repository, now


def main():
    random = np.random.default_rng(73)
    count = 360
    attendance = random.uniform(35, 100, count).round(1)
    grades = random.uniform(2, 10, count).round(1)
    failures = random.integers(0, 5, count)
    engagement = random.integers(0, 30, count)
    chance = 1 / (1 + np.exp(-(-0.06 * (attendance - 65) - .5 * (grades - 6) + .35 * failures - .03 * engagement)))
    frame = pd.DataFrame({"matricula": [f"DEMO{i:04}" for i in range(count)], "frequencia": attendance,
        "media": grades, "reprovacoes": failures, "acessos_semanais": engagement,
        "curso": random.choice(["Administração", "Tecnologia", "Pedagogia"], count),
        "sexo": random.choice(["Feminino", "Masculino"], count),
        "situacao": np.where(random.random(count) < chance, "Evadido", "Ativo")})
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ASSET_DIR / "historical_students.csv", index=False, encoding="utf-8-sig")
    current = frame.drop(columns="situacao").sample(30, random_state=12).reset_index(drop=True)
    current.to_csv(ASSET_DIR / "current_students.csv", index=False, encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="fairdrop-demo-") as directory:
        repository = Repository(Path(directory), "demo")
        store = ArtifactStore(repository)
        repository.put("datasets", {"id": DEMO_DATASET_ID, "name": "Histórico fictício de demonstração", "created_at": now(), "profile": DatasetProfiler().profile(frame).model_dump(mode="json")})
        store.save("datasets", DEMO_DATASET_ID, "dataset.joblib", frame)
        for algorithm, title in [("logistic", "Exemplo pronto · acompanhamento acadêmico"), ("rf", "Exemplo pronto · comparação de risco")]:
            card = train_and_save(repository, TrainingConfig(dataset_id=DEMO_DATASET_ID, target="situacao", positive_class="Evadido",
                name=title, algorithms=[algorithm], sensitive_features=["sexo"], identifier_column="matricula"))
            card.is_demo = True
            card.warnings.insert(0, "Modelo demonstrativo treinado exclusivamente com dados fictícios. Use para aprender o fluxo, não para decisões reais.")
            folder = ASSET_DIR / algorithm
            folder.mkdir(exist_ok=True)
            (folder / "metadata.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
            shutil.copyfile(store.path("models", card.id, "model.joblib"), folder / "model.joblib")
        repository.engine.dispose()
    print("Dois modelos e duas bases fictícias gerados em demo_assets.")


if __name__ == "__main__":
    main()
