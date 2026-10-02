import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.platform import Explanation, ExplanationFactor, TrainingConfig
from app.services.automl import AutoMLService
from app.services.explanation_context import build_context, enrich_explanation
from app.services.bias_mitigation import ReweighedClassifier
from app.services.preprocessing import build_pipeline
from sklearn.linear_model import LogisticRegression
from test_training_pipeline import students  # noqa: F401


def test_context_explains_observed_patterns_without_inventing_causes():
    frame = pd.DataFrame({"frequencia": np.arange(100), "sexo": ["F", "M"] * 50})
    y = pd.Series([1] * 25 + [0] * 75)
    result = enrich_explanation(Explanation(status="available", factors=[ExplanationFactor(feature="frequencia", value="10", impact=.2, direction="increase")]), build_context(frame, y))
    factor = result.factors[0]
    assert "mediana histórica: 49,5" in factor.historical_context
    assert "100%" in factor.historical_context and "contra 0%" in factor.historical_context
    assert factor.scenario_allowed and factor.tutor_actions
    assert "não presuma" in " ".join(factor.tutor_actions)
    protected = enrich_explanation(Explanation(status="available", factors=[ExplanationFactor(feature="sexo", value="F", impact=.1, direction="increase")]), build_context(frame, y))
    assert not protected.factors[0].scenario_allowed
    assert "característica pessoal" in " ".join(protected.factors[0].tutor_actions)


def test_reference_sample_does_not_fabricate_outcome_rates():
    frame = pd.DataFrame({"nota": range(30)})
    result = enrich_explanation(Explanation(status="available", factors=[ExplanationFactor(feature="nota", value="4", impact=.1, direction="increase")]), build_context(frame))
    assert "amostra de referência" in result.factors[0].historical_context
    assert "Não exibimos uma taxa" in result.factors[0].historical_context


@pytest.mark.parametrize("calibrate,folds", [(False, 0), (True, 3)])
def test_reweighing_is_evaluated_saved_and_predicts_without_sensitive_columns(students, tmp_path, calibrate, folds):
    # Mais exemplos por grupo/resultado para suportar as divisões internas.
    frame = pd.concat([students.drop(columns="matricula").assign(nota=students.nota + i * .00001) for i in range(4)], ignore_index=True)
    config = TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido", sensitive_features=["sexo"],
                            algorithms=["logistic"] if calibrate else ["logistic", "rf", "knn", "xgboost"], mitigation="reweighing", mitigation_features=["sexo"], calibrate=calibrate, cv_folds=folds)
    output = AutoMLService().train(frame, config)
    assert {candidate.variant for candidate in output.comparison} == {"baseline", "reweighing"}, output.failed
    assert output.mitigation.status == "evaluated"
    assert output.mitigation.before_metrics is not None and output.mitigation.after_metrics is not None
    assert "sexo" not in output.features
    if not calibrate:
        assert {c.algorithm for c in output.comparison if c.variant == "reweighing"} == {"logistic", "rf", "xgboost"}, output.failed
    assert output.context["rows"] == output.split["train"]
    future = frame[output.features].head(3)
    probabilities = output.model.predict_proba(future)
    joblib.dump(output.model, tmp_path / "weighted.joblib")
    np.testing.assert_allclose(probabilities, joblib.load(tmp_path / "weighted.joblib").predict_proba(future))


def test_weighted_pipeline_roundtrip_and_missing_protected_columns_at_inference(students, tmp_path):
    model = ReweighedClassifier(build_pipeline(LogisticRegression()), ["frequencia", "nota", "curso"], ["sexo"])
    model.fit(students, (students.situacao == "Evadido").astype(int))
    future = students.drop(columns=["sexo", "situacao", "matricula"]).head(4)
    before = model.predict_proba(future)
    joblib.dump(model, tmp_path / "reweighed.joblib")
    np.testing.assert_allclose(before, joblib.load(tmp_path / "reweighed.joblib").predict_proba(future))


def test_reweighing_reduces_disparity_in_a_controlled_proxy_example():
    rng = np.random.default_rng(73)
    group = np.repeat([0, 1], 500)
    frame = pd.DataFrame({"sexo": group, "campus": group, "frequencia": rng.uniform(30, 100, len(group)),
                          "situacao": np.where(rng.random(len(group)) < .15 + .70 * group, "Evadido", "Ativo")})
    output = AutoMLService().train(frame, TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido",
        sensitive_features=["sexo"], algorithms=["logistic"], mitigation="reweighing"))
    report = output.mitigation
    assert report.status == "evaluated"
    before = abs(report.before_fairness.comparisons[0].metrics["statistical_parity_difference"])
    after = abs(report.after_fairness.comparisons[0].metrics["statistical_parity_difference"])
    assert after < before


def test_missing_groups_preserve_baseline_and_report_mitigation_failure(students):
    students.loc[::2, "sexo"] = None
    result = AutoMLService().train(students, TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido",
        sensitive_features=["sexo"], algorithms=["logistic"], mitigation="reweighing"))
    assert result.mitigation.status == "unavailable"
    assert result.mitigation.warnings and len(result.comparison) == 1


def test_tutor_review_scenario_and_mitigation_endpoints(students, isolated_storage):
    with TestClient(app) as client:
        dataset = client.post("/datasets", files={"file": ("historico.csv", students.to_csv(index=False).encode())}).json()
        card = client.post("/train-automl", json={"dataset_id": dataset["id"], "target": "situacao", "positive_class": "Evadido",
            "sensitive_features": ["sexo"], "identifier_column": "matricula", "algorithms": ["logistic"]}).json()
        batch = client.post("/predict", data={"model_id": card["id"]}, files={"file": ("atuais.csv", students.head(4).drop(columns="situacao").to_csv(index=False).encode())}).json()
        prediction = client.get(f"/predictions/{batch['id']}/students").json()["items"][0]
        identifier = prediction["id"]
        detail = client.get(f"/students/{identifier}").json()
        assert detail["explanation"]["context_version"] == 1
        factor = next(f for f in detail["explanation"]["factors"] if f["feature"] == "frequencia")
        assert "conjunto de treino" in factor["historical_context"]
        value = (factor["reference_min"] + factor["reference_max"]) / 2
        simulated = client.post(f"/students/{identifier}/scenario", json={"feature": "frequencia", "value": value})
        assert simulated.status_code == 200, simulated.text
        assert client.get(f"/students/{identifier}").json()["prediction"]["risk_probability"] == prediction["risk_probability"]
        assert client.post(f"/students/{identifier}/scenario", json={"feature": "sexo", "value": 0}).status_code == 400
        assert client.post(f"/students/{identifier}/scenario", json={"feature": "frequencia", "value": 1e9}).status_code == 400
        review = client.put(f"/students/{identifier}/review", json={"checked_actions": ["data_checked"], "conclusion": "data_issue", "note": "Conferir registro de presença."})
        assert review.status_code == 200 and review.json()["updated_at"]
        assert client.get(f"/students/{identifier}/review").json()["conclusion"] == "data_issue"
        mitigated = client.post(f"/models/{card['id']}/mitigate", json={"feature": "sexo"})
        assert mitigated.status_code == 202, mitigated.text
        run = client.get(f"/training-runs/{mitigated.json()['id']}").json()
        assert run["status"] == "completed", run
        assert client.get(f"/models/{run['model_id']}").json()["mitigation"]["status"] == "evaluated"
        assert client.get(f"/models/{card['id']}").status_code == 200
        client.delete(f"/predictions/{batch['id']}")
        assert isolated_storage.list("tutor_reviews") == []
