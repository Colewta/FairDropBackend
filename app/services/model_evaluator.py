import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from app.schemas.platform import Metrics


def evaluate_model(model, X, y, threshold: float) -> tuple[Metrics, np.ndarray]:
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    observed, predicted = calibration_curve(y, probability, n_bins=8, strategy="quantile")
    both_classes = len(np.unique(y)) == 2
    return Metrics(
        accuracy=accuracy_score(y, prediction), precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0), f1=f1_score(y, prediction, zero_division=0),
        roc_auc=roc_auc_score(y, probability) if both_classes else None,
        pr_auc=average_precision_score(y, probability) if both_classes else None,
        specificity=float(tn / (tn + fp)) if tn + fp else 0,
        balanced_accuracy=balanced_accuracy_score(y, prediction),
        brier_score=brier_score_loss(y, probability), confusion_matrix=matrix.tolist(),
        calibration_curve=[{"predicted": float(p), "observed": float(o)} for p, o in zip(predicted, observed)],
    ), prediction
