from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.dataset import ColumnRole as Role, DatasetError
from app.services.dataset_profiler import DatasetProfiler, infer_type
from app.services.dataset_reader import read_dataset
from app.services.leakage_detector import LeakageDetector
from app.services.semantic_analyzer import LocalSemanticAnalyzer


@pytest.fixture
def historical():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "matricula": [f"RA{i:04}" for i in range(100)],
        "situacao_final": ["Ativo", "Evadido"] * 50,
        "sexo": ["F", "F", "M", "M"] * 25,
        "idade": rng.integers(18, 60, 100),
        "frequencia": rng.uniform(0, 100, 100),
        "data_cancelamento": [None, "2025-01-01"] * 50,
        "constante": [1] * 100,
    })


def test_profile_roles_features_and_immutability(historical):
    before = historical.copy(deep=True)
    result = DatasetProfiler().profile(historical)
    assert result.selected_target == "situacao_final"
    assert result.target_source == "suggested"
    assert result.potential_ids == ["matricula"]
    assert set(result.fairness_features) == {"sexo", "idade"}
    assert result.prediction_features == ["frequencia"]
    assert result.missing_cells_percentage == pytest.approx(7.143)
    assert any(a.column == "data_cancelamento" and a.risk == "HIGH" for a in result.leakage_alerts)
    pd.testing.assert_frame_equal(historical, before)
    assert "RA0001" not in result.model_dump_json()


def test_semantics_use_tokens_and_can_be_extended():
    names = ["score", "cor", "carga", "renda_familiar", "dataNascimento", "student_id"]
    result = LocalSemanticAnalyzer().analyze_columns(names)
    assert Role.SENSITIVE not in result["score"]
    assert Role.IDENTIFIER not in result["carga"]
    assert Role.SENSITIVE in result["cor"]
    assert Role.FINANCIAL in result["renda_familiar"]
    assert Role.SENSITIVE in result["dataNascimento"]
    assert Role.IDENTIFIER in result["student_id"]
    assert LocalSemanticAnalyzer({Role.ENGAGEMENT: {"clicks"}}).analyze_columns(["clicks"])["clicks"] == [Role.ENGAGEMENT]


@pytest.mark.parametrize("values,expected", [
    (["sim", "não", None], "boolean"), (["2025-01-01", "2025-02-01"], "date"),
    (["2.5", "3.2", None], "numeric"), (["azul", "verde"], "categorical"),
    ([None, None], "unknown"),
])
def test_types(values, expected):
    assert infer_type(pd.Series(values)) == expected


def test_identifiers_do_not_include_continuous_measurements():
    df = pd.DataFrame({"codigo": [f"ST{i:03}" for i in range(40)],
                       "medida": np.linspace(1, 10, 40), "nota": range(40)})
    result = DatasetProfiler().profile(df)
    assert result.potential_ids == ["codigo"]
    assert result.selected_target is None
    assert result.fairness_features == []


def test_target_override_and_invalid_target(historical):
    result = DatasetProfiler().profile(historical, "sexo")
    assert result.selected_target == "sexo"
    assert result.target_source == "user"
    assert "sexo" not in result.fairness_features
    assert any(a.column == "situacao_final" for a in result.leakage_alerts)
    with pytest.raises(DatasetError, match="não existe"):
        DatasetProfiler().profile(historical, "inexistente")


def test_leakage_copy_correlation_and_unique_categories():
    df = pd.DataFrame({"target": [0, 1] * 30, "copy": ["A", "B"] * 30,
                       "numeric": np.tile([0.001, 0.999], 30) + np.arange(60) * 0.00001,
                       "unique": [f"text{i}" for i in range(60)]})
    alerts = {a.column: a.risk for a in LeakageDetector().detect(df, "target", [])}
    assert alerts == {"copy": "HIGH", "numeric": "MEDIUM"}


def test_health_missing_constant_duplicates_imbalance():
    df = pd.DataFrame({"status": ["Ativo"] * 98 + ["Evadido"] * 2,
                       "vazio": [None] * 100, "infinito": [np.inf] * 100})
    result = DatasetProfiler().profile(df)
    codes = {issue.code for issue in result.health.issues}
    assert {"MISSING_VALUES", "CONSTANT_COLUMN", "DUPLICATES", "CLASS_IMBALANCE", "NON_FINITE_VALUES"} <= codes
    assert 'Infinity' not in result.model_dump_json()
    assert 'NaN' not in result.model_dump_json()
    one_class = DatasetProfiler().profile(df.iloc[:30], "status")
    assert "TARGET_SINGLE_CLASS" in {i.code for i in one_class.health.issues}


@pytest.mark.parametrize("delimiter,encoding", [(";", "cp1252"), ("\t", "utf-8-sig"), ("|", "utf-16"), (",", "utf-8")])
def test_csv_encodings_delimiters_and_leading_zeros(delimiter, encoding):
    content = f"matricula{delimiter}situação\n001{delimiter}Ativo\n002{delimiter}Evadido\n".encode(encoding)
    df = read_dataset(content, "base.csv")
    assert df.columns.tolist() == ["matricula", "situação"]
    assert df.iloc[0, 0] == "001"


@pytest.mark.parametrize("content", [b"a,b\n1,2,3\n", b"", b"a,b\n"])
def test_bad_csv_rejected(content):
    with pytest.raises(DatasetError):
        read_dataset(content, "base.csv")


def test_headers_are_repaired_with_visible_warning():
    df = read_dataset(b"a,a,\n1,2,3\n", "base.csv")
    assert df.columns.tolist() == ["a", "a_2", "coluna_3"]
    assert len(df.attrs["import_warnings"]) == 2


def test_regional_numbers_and_target_detection():
    result = DatasetProfiler().profile(pd.DataFrame({"media": ["1,5", "2,5"], "marital-status": ["married", "single"], "y": ["yes", "no"]}))
    assert result.selected_target == "y"
    assert result.column_profiles[0].mean == 2
    assert "marital-status" not in result.potential_targets
    adult = DatasetProfiler().profile(pd.DataFrame({"income": ["<=50K", ">50K"], "age": [30, 40]}))
    assert adult.selected_target == "income"


def test_renamed_excel_with_two_header_rows():
    import xlwt
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("data")
    for row, values in enumerate([["", "X1", "Y"], ["ID", "AGE", "default payment next month"], [1, 24, 0], [2, 35, 1]]):
        for col, value in enumerate(values):
            sheet.write(row, col, value)
    output = BytesIO(); workbook.save(output)
    df = read_dataset(output.getvalue(), "renamed.csv")
    assert df.shape == (2, 3)
    assert df.columns.tolist() == ["ID", "AGE", "default payment next month"]
    assert df.attrs["import_info"]["format"] == "XLS"
    assert len(df.attrs["import_warnings"]) == 2


def test_one_column_csv():
    assert read_dataset(b"status\nAtivo\nEvadido\n", "base.csv").shape == (2, 1)


def test_reader_limits():
    with patch("app.services.dataset_reader.MAX_UPLOAD_BYTES", 2), pytest.raises(DatasetError):
        read_dataset(b"a,b\n1,2", "base.csv")
    with patch("app.services.dataset_reader.MAX_ROWS", 1), pytest.raises(DatasetError):
        read_dataset(b"a,b\n1,2\n3,4", "base.csv")
    with patch("app.services.dataset_reader.MAX_COLUMNS", 1), pytest.raises(DatasetError):
        read_dataset(b"a,b\n1,2", "base.csv")


@pytest.mark.parametrize("extension", ["xls", "xlsx"])
def test_corrupt_excel_is_client_error(extension):
    with TestClient(app) as client:
        response = client.post("/analyze-dataset", files={"file": (f"base.{extension}", b"invalid workbook")})
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "INVALID_FILE"


def test_excel_formats(historical):
    xlsx = BytesIO()
    historical.to_excel(xlsx, index=False)
    assert read_dataset(xlsx.getvalue(), "base.xlsx").shape == historical.shape
    import xlwt
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("alunos")
    for row, values in enumerate([["matricula", "status"], ["001", "Ativo"], ["002", "Evadido"]]):
        for col, value in enumerate(values):
            sheet.write(row, col, value)
    xls = BytesIO()
    workbook.save(xls)
    df = read_dataset(xls.getvalue(), "base.xls")
    assert df.iloc[0, 0] == "001"


def test_api_upload_analyze_override_and_legacy_routes(historical):
    with TestClient(app) as client:
        content = historical.to_csv(index=False).encode()
        response = client.post("/analyze-dataset", files={"file": ("base.csv", content, "text/csv")})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["selected_target"] == "situacao_final"
        assert body["prediction_features"] == ["frequencia"]
        override = client.post("/analyze-dataset", data={"target": "sexo"}, files={"file": ("base.csv", content)})
        assert override.json()["selected_target"] == "sexo"
        invalid = client.post("/analyze-dataset", data={"target": "ausente"}, files={"file": ("base.csv", content)})
        assert invalid.status_code == 400
        assert invalid.json()["detail"]["error"] == "TARGET_NOT_FOUND"
        invalid_type = client.post("/analyze-dataset", files={"file": ("base.exe", b"abc")})
        assert invalid_type.status_code == 400
        assert client.get("/health").json() == {"status": "ok"}
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/analyze", "/train", "/simulate", "/analyze-dataset"} <= paths.keys()


def test_api_unexpected_failure_hides_personal_data():
    with TestClient(app) as client, patch("app.routes.datasets.read_upload", side_effect=RuntimeError("CPF privado")):
        response = client.post("/analyze-dataset", files={"file": ("base.csv", b"a\n1")})
        assert response.status_code == 500
        assert "CPF" not in response.text
