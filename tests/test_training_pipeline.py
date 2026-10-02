import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from app.schemas.dataset import DatasetError
from app.schemas.platform import TrainingConfig
from app.services.automl import AutoMLService
from app.services.fairness_evaluator import FairnessEvaluator
from app.services.model_selector import ModelSelector
from app.services.preprocessing import build_pipeline


@pytest.fixture
def students():
    rng = np.random.default_rng(9)
    attendance = rng.uniform(30, 100, 240)
    outcome = np.where(attendance + rng.normal(0, 30, 240) < 70, "Evadido", "Ativo")
    return pd.DataFrame({"matricula": [f"RA{i:04}" for i in range(240)], "frequencia": attendance,
                         "nota": rng.normal(6, 2, 240), "curso": ["A", "B", "C"] * 80,
                         "sexo": ["F", "M"] * 120, "situacao": outcome})


def test_pipeline_serialization_unknown_categories_and_train_only_imputation(tmp_path):
    frame = pd.DataFrame({"nota": [1., 3., np.nan, 5.], "curso": ["A", "B", None, "A"]})
    model = build_pipeline(LogisticRegression()).fit(frame, [0, 1, 0, 1])
    assert model["columns"].named_transformers_["numeric"]["imputer"].statistics_[0] == 3
    future = pd.DataFrame({"extra": [999], "curso": ["novo"], "nota": [np.nan]})
    before = model.predict_proba(future)
    joblib.dump(model, tmp_path / "pipeline.joblib")
    assert np.allclose(before, joblib.load(tmp_path / "pipeline.joblib").predict_proba(future))
    with pytest.raises(DatasetError, match="nota"):
        model.predict_proba(future.drop(columns="nota"))


def test_automl_same_split_and_sensitive_exclusion(students):
    config = TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido", sensitive_features=["sexo"], identifier_column="matricula")
    output = AutoMLService().train(students, config)
    assert len(output.comparison) == 4, output.failed
    assert "matricula" not in output.features and "sexo" not in output.features
    assert sum(output.split.values()) == len(students)
    assert output.metrics.roc_auc is not None
    assert all(sum(sum(row) for row in c.validation.confusion_matrix) == output.split["validation"] for c in output.comparison)


def test_calibration_and_cv(students):
    config = TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido", algorithms=["logistic"], calibrate=True, cv_folds=3)
    output = AutoMLService().train(students, config)
    assert "recall" in output.comparison[0].cross_validation
    assert output.metrics.calibration_curve


def test_invalid_weights_and_repeated_students(students):
    with pytest.raises(DatasetError):
        ModelSelector("best_balanced", {"recall": -1})
    students.loc[1, "matricula"] = students.loc[0, "matricula"]
    with pytest.raises(DatasetError, match="múltiplos"):
        AutoMLService().train(students, TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido", identifier_column="matricula"))


def test_fairness_known_disparity_and_missing_groups():
    df = pd.DataFrame({"sexo": ["F"] * 20 + ["M"] * 20})
    y = pd.Series([0, 1] * 20)
    evaluator = FairnessEvaluator(["sexo"], {"sexo": "M"}).fit(df)
    report = evaluator.evaluate(df, y, np.array([0] * 20 + [1] * 20))
    assert report.status == "relevant_disparity"
    assert report.comparisons[0].metrics["statistical_parity_difference"] == -1
    assert evaluator.evaluate(df.iloc[:20], y.iloc[:20], np.zeros(20)).status == "unavailable"


def test_numeric_target_uses_confirmed_rule(students):
    students["renda"] = np.linspace(1000, 100000, len(students))
    config = TrainingConfig(dataset_id="test", target="renda", positive_class="Renda acima de 50 mil",
                            target_rule={"operator": "ge", "value": 50000}, algorithms=["logistic"])
    result = AutoMLService().train(students, config)
    assert result.classes == ["Demais valores", "Renda acima de 50 mil"]
    assert "renda" not in result.features


def test_feature_selection_does_not_learn_from_holdout(students):
    from app.core.config import RANDOM_STATE
    y = (students["situacao"] == "Evadido").astype(int)
    trainval, test = train_test_split(students.index, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    train, validation = train_test_split(trainval, test_size=.25, stratify=y.loc[trainval], random_state=RANDOM_STATE)
    students["observacao"] = 0
    students.loc[list(test) + list(validation), "observacao"] = range(len(test) + len(validation))
    config = TrainingConfig(dataset_id="test", target="situacao", positive_class="Evadido", algorithms=["logistic"])
    output = AutoMLService().train(students, config)
    assert "observacao" not in output.features
    assert output.split["train"] == len(train)
