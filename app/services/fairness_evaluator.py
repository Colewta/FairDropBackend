import logging

import numpy as np
import pandas as pd

from app.schemas.platform import FairnessComparison, FairnessReport
from app.services.fairness import avaliar_fairness_aif360

logger = logging.getLogger(__name__)


def group_rates(y: pd.Series, prediction: np.ndarray, mask: pd.Series) -> dict:
    actual = y.loc[mask].to_numpy()
    predicted = prediction[mask.to_numpy()]
    positive, negative = actual == 1, actual == 0
    return {"alert_rate": float(predicted.mean()),
            "recall": float(predicted[positive].mean()) if positive.any() else None,
            "false_positive_rate": float(predicted[negative].mean()) if negative.any() else None,
            "false_negative_rate": float(1 - predicted[positive].mean()) if positive.any() else None,
            "actual_positive_count": int(positive.sum()), "actual_negative_count": int(negative.sum())}


class FairnessEvaluator:
    """Grupos definidos no treino; comparação exclusivamente em validação/teste."""
    def __init__(self, features: list[str], references: dict[str, str]):
        self.features = features
        self.references = references
        self.groups: dict[str, dict] = {}

    def fit(self, frame: pd.DataFrame):
        for feature in self.features:
            series = frame[feature].dropna()
            numeric = pd.to_numeric(series, errors="coerce")
            if series.nunique() > 12 and numeric.notna().all() and len(numeric):
                cut = float(numeric.median())
                groups = pd.Series(np.where(numeric <= cut, "até mediana", "acima da mediana"))
            elif series.nunique() > 12:
                self.groups[feature] = {"unsupported": True}
                continue
            else:
                cut = None
                groups = series.astype(str)
            reference = self.references.get(feature)
            if reference is None and len(groups):
                reference = str(groups.mode().iloc[0])
            self.groups[feature] = {"cut": cut, "reference": reference,
                                    "automatic": feature not in self.references}
        return self

    def evaluate(self, frame: pd.DataFrame, y: pd.Series, prediction: np.ndarray) -> FairnessReport:
        comparisons = []
        warnings = []
        for feature, spec in self.groups.items():
            if spec.get("unsupported"):
                warnings.append(f"{feature}: mais de 12 grupos; defina categorias antes da avaliação.")
                continue
            series = frame[feature]
            if spec["cut"] is not None:
                numeric = pd.to_numeric(series, errors="coerce")
                groups = pd.Series(np.where(numeric <= spec["cut"], "até mediana", "acima da mediana"), index=frame.index).where(numeric.notna())
                warnings.append(f"{feature}: agrupado pela mediana do treino ({spec['cut']:.4g}).")
            else:
                groups = series.map(lambda v: None if pd.isna(v) else str(v))
            reference = spec["reference"]
            if spec["automatic"]:
                warnings.append(f"{feature}: referência descritiva = grupo mais frequente no treino; não significa privilégio social.")
            if groups.isna().any():
                warnings.append(f"{feature}: {int(groups.isna().sum())} registros sem grupo foram omitidos somente da avaliação de fairness.")
            if reference not in set(groups.dropna()):
                warnings.append(f"{feature}: grupo de referência ausente neste conjunto.")
                continue
            for group in sorted(set(groups.dropna()) - {reference}):
                mask = groups.isin([group, reference])
                group_count = int((groups == group).sum())
                ref_count = int((groups == reference).sum())
                if min(group_count, ref_count) < 5:
                    warnings.append(f"{feature}: comparação omitida por grupo com menos de 5 registros.")
                    continue
                data = pd.DataFrame({"sensitive": (groups[mask] == reference).astype(int), "label": y.loc[mask].astype(int)})
                try:
                    metrics = avaliar_fairness_aif360(data, y.loc[mask], prediction[mask.to_numpy()], "label", "sensitive")
                except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
                    logger.warning("Fairness indisponível: %s", type(exc).__name__)
                    warnings.append(f"{feature}: não foi possível calcular esta comparação.")
                    continue
                differences = [abs(metrics[key]) for key in ("statistical_parity_difference", "equal_opportunity_difference", "average_odds_difference") if metrics[key] is not None]
                di = metrics["disparate_impact"]
                relevant = any(v > 0.2 for v in differences) or (di is not None and not 0.65 <= di <= 1.5)
                attention = any(v > 0.1 for v in differences) or (di is not None and not 0.8 <= di <= 1.25)
                unavailable = any(metrics[k] is None for k in metrics if k != "fairness_score")
                status = "relevant_disparity" if relevant else "attention" if attention or unavailable else "low_signal"
                comparisons.append(FairnessComparison(feature=feature, group=group, reference=reference,
                    group_count=group_count, reference_count=ref_count, metrics=metrics, status=status,
                    group_rates=group_rates(y, prediction, groups == group), reference_rates=group_rates(y, prediction, groups == reference),
                    explanation="Diferenças entre grupos exigem revisão do contexto e da amostra; os critérios não certificam justiça."))
        scores = [c.metrics["fairness_score"] for c in comparisons if c.metrics.get("fairness_score") is not None]
        status = "unavailable" if not comparisons else "relevant_disparity" if any(c.status == "relevant_disparity" for c in comparisons) else "attention" if warnings or any(c.status == "attention" for c in comparisons) else "low_signal"
        return FairnessReport(status=status, comparisons=comparisons, warnings=warnings,
                              score=min(scores) if scores else None)
