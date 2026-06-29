import math

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def calcular_score_fairness(metricas):
    componentes = []

    for chave in (
        "statistical_parity_difference",
        "equal_opportunity_difference",
        "average_odds_difference",
    ):
        valor = _safe_float(metricas.get(chave))
        if valor is not None:
            componentes.append(max(0.0, 1.0 - abs(valor)))

    disparate_impact = _safe_float(metricas.get("disparate_impact"))
    if disparate_impact is not None:
        componentes.append(max(0.0, 1.0 - abs(1.0 - disparate_impact)))

    if not componentes:
        return None

    return round(sum(componentes) / len(componentes), 4)


def avaliar_fairness_aif360(df_original, y_true, y_pred, target, sensitive):
    df = df_original.copy()

    df[target] = y_true.values
    df["prediction"] = y_pred

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes

    df[target] = pd.to_numeric(df[target], errors="raise")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="raise")
    df[sensitive] = pd.to_numeric(df[sensitive], errors="raise")

    non_numeric_columns = [
        col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_columns:
        raise ValueError(
            "Columns must be numerical after preprocessing: "
            + ", ".join(non_numeric_columns)
        )

    dataset_true = BinaryLabelDataset(
        df=df.drop(columns=["prediction"]),
        label_names=[target],
        protected_attribute_names=[sensitive],
    )

    df_pred = df.drop(columns=["prediction"]).copy()
    df_pred[target] = df["prediction"]

    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=[target],
        protected_attribute_names=[sensitive],
    )

    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=[{sensitive: 0}],
        privileged_groups=[{sensitive: 1}],
    )

    metricas = {
        "statistical_parity_difference": _safe_float(metric.statistical_parity_difference()),
        "disparate_impact": _safe_float(metric.disparate_impact()),
        "equal_opportunity_difference": _safe_float(metric.equal_opportunity_difference()),
        "average_odds_difference": _safe_float(metric.average_odds_difference()),
    }
    metricas["fairness_score"] = calcular_score_fairness(metricas)

    return metricas
