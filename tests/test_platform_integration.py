from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_registry import ModelRegistry
from app.services.prediction import risk_level
from app.services.repository import Repository
from test_training_pipeline import students  # noqa: F401


def test_upload_train_save_reload_predict_explain_delete(students, isolated_storage):
    with TestClient(app) as client:
        response = client.post("/datasets", files={"file": ("historico.csv", students.to_csv(index=False).encode())})
        assert response.status_code == 201, response.text
        dataset = response.json()
        config = {"dataset_id": dataset["id"], "target": "situacao", "positive_class": "Evadido",
                  "sensitive_features": ["sexo"], "identifier_column": "matricula", "algorithms": ["logistic", "rf"]}
        response = client.post("/training-runs", json=config)
        assert response.status_code == 202, response.text
        run = client.get(f"/training-runs/{response.json()['id']}").json()
        assert run["status"] == "completed", run
        card = client.get(f"/models/{run['model_id']}").json()
        assert card["explanation"]["status"] == "available", card["explanation"]
        fresh_repository = Repository(isolated_storage.root, isolated_storage.institution_id)
        _, artifact = ModelRegistry(fresh_repository).load(card["id"])
        future = students.drop(columns="situacao").iloc[:12].copy()
        future["curso"] = "Categoria nova"
        future["turma"] = ["A" if i % 2 else "B" for i in range(len(future))]
        future = future[future.columns[::-1]]
        before = artifact["pipeline"].predict_proba(future)[:, 1]
        response = client.post("/predict", data={"model_id": card["id"]}, files={"file": ("atuais.csv", future.to_csv(index=False).encode())})
        assert response.status_code == 201, response.text
        batch = response.json()
        assert batch["total"] == 12 and sum(batch["summary"].values()) == 12
        page = client.get(f"/predictions/{batch['id']}/students?page_size=5").json()
        assert page["total"] == 12 and len(page["items"]) == 5
        assert batch["group_filters"] == {"turma": ["A", "B"], "curso": ["Categoria nova"]}
        filtered = client.get(f"/predictions/{batch['id']}/students", params={"group_column": "turma", "group_value": "A"}).json()
        assert filtered["total"] == 6 and all(item["groups"]["turma"] == "A" for item in filtered["items"])
        assert client.get(f"/predictions/{batch['id']}/students", params={"group_column": "nome", "group_value": "x"}).status_code == 400
        student = client.get(f"/students/{page['items'][0]['id']}").json()
        explanation = student["explanation"]
        assert set(student["used_information"]) == set(card["features"])
        assert "turma" in student["ignored_columns"] and "matricula" in student["ignored_columns"]
        assert explanation["status"] == "available", explanation
        assert np.isclose(explanation["base_value"] + sum(f["impact"] for f in explanation["factors"]), explanation["probability"])
        expected = before[future["matricula"].tolist().index(student["prediction"]["student_id"])]
        assert np.isclose(student["prediction"]["risk_probability"], expected)
        with patch("app.services.prediction.SHAPExplainer.explain", side_effect=AssertionError("cache not used")):
            assert client.get(f"/students/{page['items'][0]['id']}").status_code == 200
        missing = client.post("/predict", data={"model_id": card["id"]}, files={"file": ("bad.csv", b"matricula\nRA1\n")})
        assert missing.status_code == 400 and missing.json()["detail"]["error"] == "MISSING_FEATURES"
        assert client.get("/dashboard").json()["latest_batch"]["id"] == batch["id"]
        assert client.delete(f"/models/{card['id']}").status_code == 204
        assert client.get(f"/models/{card['id']}").status_code == 404
        assert client.delete(f"/predictions/{batch['id']}").status_code == 204
        assert client.delete(f"/datasets/{dataset['id']}").status_code == 204
        fresh_repository.engine.dispose()


def test_threshold_boundaries():
    assert risk_level(0.299, 0.3, 0.7) == "low"
    assert risk_level(0.3, 0.3, 0.7) == "medium"
    assert risk_level(0.7, 0.3, 0.7) == "high"


def test_institution_isolation(isolated_storage):
    isolated_storage.put("datasets", {"id": "a" * 32, "created_at": "today"})
    other = Repository(isolated_storage.root, "other")
    assert other.list("datasets") == []
    other.engine.dispose()


def test_bundled_models_and_one_click_demo(isolated_storage):
    from app.services.demo import install_demo
    install_demo(isolated_storage)
    with TestClient(app) as client:
        examples = client.get("/demo").json()
        assert len(examples["models"]) == 2
        assert all(model["is_demo"] for model in examples["models"])
        assert client.get("/demo/files/current").status_code == 200
        response = client.post("/demo/predict")
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 30
        model_id = examples["models"][0]["id"]
        client.delete(f"/models/{model_id}")
        install_demo(isolated_storage)
        assert len(client.get("/demo").json()["models"]) == 1


def test_legacy_routes_use_saved_pipeline(students):
    with TestClient(app) as client:
        response = client.post("/train", data={"target": "situacao", "sensitive": "sexo", "model_type": "logistic"}, files={"file": ("history.csv", students.to_csv(index=False).encode())})
        assert response.status_code == 200, response.text
        legacy = response.json()
        assert {"metricas", "fairness", "modelos", "analise_dataset", "preprocessamento"} <= legacy.keys()
        result = client.post("/simulate", json={"model_id": legacy["model_id"], "frequencia": 60, "nota": 5, "curso": "A"})
        assert result.status_code == 200, result.text
        assert 0 <= result.json()["probabilidade_evasao"] <= 1
