import logging

import numpy as np
import pandas as pd

from app.core.config import RANDOM_STATE, SHAP_BACKGROUND_ROWS, SHAP_GLOBAL_ROWS, SHAP_MAX_FEATURES
from app.schemas.platform import Explanation, ExplanationFactor

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def explain(self, model, background: pd.DataFrame, rows: pd.DataFrame, global_mode: bool = False) -> Explanation:
        if len(background.columns) > SHAP_MAX_FEATURES:
            return Explanation(status="unavailable", message=f"SHAP limitado a {SHAP_MAX_FEATURES} variáveis nesta instalação.")
        try:
            import shap
            columns = list(background.columns)
            sample = background.head(SHAP_BACKGROUND_ROWS)
            evaluation = rows[columns].head(SHAP_GLOBAL_ROWS if global_mode else 1)
            # Máscara numérica representa cada valor bruto, inclusive categorias e ausências.
            # O modelo continua recebendo o DataFrame original, pelo pipeline completo.
            categories = {}
            for column in columns:
                values = pd.concat([sample[column], evaluation[column]]).map(lambda v: None if pd.isna(v) else str(v))
                categories[column] = list(dict.fromkeys(values.tolist()))

            def encode(frame):
                return np.column_stack([[categories[c].index(None if pd.isna(v) else str(v)) for v in frame[c]] for c in columns]).astype(float)

            def predict(encoded):
                raw = pd.DataFrame({c: [categories[c][int(round(row[i]))] for row in encoded] for i, c in enumerate(columns)})
                return model.predict_proba(raw)[:, 1]

            explainer = shap.Explainer(predict, encode(sample), algorithm="permutation", feature_names=columns, seed=RANDOM_STATE)
            values = explainer(encode(evaluation), max_evals=2 * len(columns) + 1, silent=True)
            if global_mode:
                return Explanation(status="available", global_importance={c: float(v) for c, v in zip(columns, np.abs(values.values).mean(axis=0))},
                                   message=f"Média do impacto absoluto em {len(evaluation)} exemplos de treino. Aproximação SHAP, sem interpretação causal.")
            factors = [ExplanationFactor(feature=c, value=None if pd.isna(evaluation.iloc[0][c]) else str(evaluation.iloc[0][c]),
                                         impact=float(v), direction="increase" if v >= 0 else "decrease") for c, v in zip(columns, values.values[0])]
            factors.sort(key=lambda f: -abs(f.impact))
            return Explanation(status="available", base_value=float(values.base_values[0]), probability=float(model.predict_proba(evaluation)[0, 1]), factors=factors, reference_rows=len(sample))
        except Exception as exc:
            logger.warning("SHAP indisponível: %s", type(exc).__name__)
            return Explanation(status="unavailable", message="Não foi possível calcular SHAP para este modelo. A probabilidade continua disponível.")
