import math

from app.core.config import MODEL_SELECTION_WEIGHTS
from app.schemas.dataset import DatasetError
from app.schemas.platform import FairnessReport, Metrics


class ModelSelector:
    def __init__(self, strategy: str, weights: dict[str, float] | None = None):
        self.weights = MODEL_SELECTION_WEIGHTS[strategy].copy() if weights is None else weights
        allowed = {"recall", "roc_auc", "pr_auc", "fairness", "calibration"}
        if (not self.weights or set(self.weights) - allowed or
                any(not math.isfinite(w) or w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0):
            raise DatasetError("INVALID_WEIGHTS", "Os pesos precisam ser finitos, não negativos e somar mais de zero.")

    def score(self, metrics: Metrics, fairness: FairnessReport):
        components = {"recall": metrics.recall, "roc_auc": metrics.roc_auc, "pr_auc": metrics.pr_auc,
                      "fairness": fairness.score, "calibration": 1 - metrics.brier_score}
        available = {key: value for key, value in components.items() if value is not None and self.weights.get(key, 0) > 0}
        total = sum(self.weights[key] for key in available)
        if total == 0:
            raise DatasetError("SELECTION_UNAVAILABLE", "Nenhum critério de seleção está disponível com os pesos escolhidos.")
        weights = {key: self.weights[key] / total for key in available}
        return sum(available[key] * weights[key] for key in available), available, weights
