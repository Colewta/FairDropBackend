"""Reponderação AIF360 ajustada dentro de cada fold, sem alterar dados de teste."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from app.schemas.dataset import DatasetError
from app.services.numeric import parse_numeric


class ReweighedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, features, sensitive_features):
        self.estimator = estimator
        self.features = features
        self.sensitive_features = sensitive_features

    def fit(self, X, y):
        from aif360.algorithms.preprocessing import Reweighing
        from aif360.datasets import BinaryLabelDataset
        if not self.sensitive_features:
            raise DatasetError("MITIGATION_GROUP_REQUIRED", "Confirme ao menos um grupo para comparar a mitigação.")
        groups = pd.DataFrame(index=X.index)
        for column in self.sensitive_features:
            series = X[column]
            if series.isna().any():
                raise DatasetError("MITIGATION_MISSING_GROUP", f"{column}: preencha ou revise os grupos ausentes antes da reponderação.")
            numeric = parse_numeric(series)
            if series.nunique() > 12 and numeric.notna().all():
                groups[column] = np.where(numeric <= numeric.median(), "até mediana", "acima da mediana")
            elif series.nunique() <= 12:
                groups[column] = series.astype(str)
            else:
                raise DatasetError("MITIGATION_TOO_MANY_GROUPS", f"{column}: defina até 12 grupos antes da mitigação.")
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(groups))
        counts = pd.crosstab(codes, np.asarray(y)).reindex(columns=[0, 1], fill_value=0)
        if len(counts) < 2 or len(counts) > 32 or (counts < 5).any().any():
            raise DatasetError("MITIGATION_SMALL_GROUPS", "Cada grupo precisa de pelo menos 5 exemplos de cada resultado no treino de cada divisão; revise a amostra e os grupos.")
        # A API sklearn importa FairAdapt/R mesmo sem utilizá-lo em instalações
        # antigas. A API clássica do AIF360 evita esse extra. Para vários grupos,
        # preservamos somente os pesos do grupo focal de cada chamada; isso aplica
        # a mesma fórmula N(grupo)*N(classe)/(N*N(grupo,classe)) a cada célula.
        weights = np.empty(len(X), dtype=float)
        for code in counts.index:
            mask = codes == code
            data = BinaryLabelDataset(df=pd.DataFrame({"group": mask.astype(int), "label": np.asarray(y)}),
                                      label_names=["label"], protected_attribute_names=["group"])
            weighted = Reweighing(unprivileged_groups=[{"group": 0}], privileged_groups=[{"group": 1}]).fit_transform(data)
            weights[mask] = weighted.instance_weights[mask]
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise DatasetError("MITIGATION_INVALID_WEIGHTS", "Não foi possível obter pesos válidos para estes grupos.")
        self.weight_summary_ = {"min": float(weights.min()), "max": float(weights.max()),
                                "mean": float(weights.mean()), "effective_samples": float(weights.sum() ** 2 / np.square(weights).sum()),
                                "training_samples": float(len(X)), "groups": float(len(counts))}
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[self.features], y, model__sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X[self.features])

    def predict(self, X):
        return self.estimator_.predict(X[self.features])


def core_pipeline(model):
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator
    return model.estimator_ if isinstance(model, ReweighedClassifier) else model


def weight_summary(model):
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator
    return getattr(model, "weight_summary_", {})


def mitigation_report(config, results, fitted, evaluator, test_frame, test_y, failed):
    from app.schemas.platform import MitigationReport
    from app.services.model_evaluator import evaluate_model
    if config.mitigation == "none":
        return MitigationReport()
    features = config.mitigation_features or config.sensitive_features[:1]
    warnings = [message for key, message in failed.items() if key.endswith(":reweighing")]
    paired = [item for item in results if item.variant == "reweighing" and (item.algorithm, "baseline") in fitted]
    if not paired:
        return MitigationReport(method="reweighing", status="unavailable", features=features, warnings=warnings,
                                message="Não foi possível comparar a reponderação nesta configuração. O modelo original foi preservado.")
    # A variante é escolhida pela validação. O teste serve somente à auditoria final.
    chosen = paired[0]
    before, before_prediction = evaluate_model(fitted[(chosen.algorithm, "baseline")], test_frame, test_y, config.threshold)
    after_model = fitted[(chosen.algorithm, "reweighing")]
    after, after_prediction = evaluate_model(after_model, test_frame, test_y, config.threshold)
    before_fairness = evaluator.evaluate(test_frame, test_y, before_prediction)
    after_fairness = evaluator.evaluate(test_frame, test_y, after_prediction)
    selected = results[0].variant == "reweighing"
    return MitigationReport(method="reweighing", status="evaluated", selected=selected, features=features,
        algorithm=chosen.algorithm, before_metrics=before, after_metrics=after, before_fairness=before_fairness,
        after_fairness=after_fairness, weight_summary=weight_summary(after_model), warnings=warnings,
        message="A reponderação foi selecionada pelos critérios de validação." if selected else
                "A comparação foi realizada, mas o modelo sem reponderação obteve melhor pontuação na validação e foi mantido.")
