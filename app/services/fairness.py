import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def avaliar_fairness_aif360(df_original, y_true, y_pred, target, sensitive):
    df = df_original.copy()

    # Adiciona labels reais e previstos para calcular as metricas.
    df[target] = y_true.values
    df["prediction"] = y_pred

    # Converte qualquer coluna nao numerica para codigos numericos.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes

    # Garante que os campos usados pelo AIF360 estejam em formato numerico.
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

    # Dataset real sem a coluna auxiliar de predicao.
    dataset_true = BinaryLabelDataset(
        df=df.drop(columns=["prediction"]),
        label_names=[target],
        protected_attribute_names=[sensitive]
    )

    # Dataset previsto com a predicao como label.
    df_pred = df.drop(columns=["prediction"]).copy()
    df_pred[target] = df["prediction"]

    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=[target],
        protected_attribute_names=[sensitive]
    )

    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=[{sensitive: 0}],
        privileged_groups=[{sensitive: 1}]
    )

    return {
        "statistical_parity_difference": metric.statistical_parity_difference(),
        "disparate_impact": metric.disparate_impact(),
        "equal_opportunity_difference": metric.equal_opportunity_difference(),
        "average_odds_difference": metric.average_odds_difference()
    }
