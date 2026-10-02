import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from app.core.config import RANDOM_STATE
from app.schemas.dataset import ColumnRole, DatasetError
from app.schemas.platform import CandidateResult, TrainingConfig, MitigationReport
from app.services.dataset_profiler import DatasetProfiler
from app.services.fairness_evaluator import FairnessEvaluator
from app.services.model_evaluator import evaluate_model
from app.services.model_selector import ModelSelector
from app.services.models import criar_modelo, extrair_importancia
from app.services.preprocessing import build_pipeline
from app.services.numeric import parse_numeric
from app.services.bias_mitigation import ReweighedClassifier, core_pipeline, mitigation_report

logger = logging.getLogger(__name__)


@dataclass
class TrainingOutput:
    model: object
    algorithm: str
    comparison: list[CandidateResult]
    failed: dict[str, str]
    features: list[str]
    feature_schema: dict[str, str]
    excluded: list[str]
    classes: list[str]
    metrics: object
    fairness: object
    feature_importance: dict[str, float]
    split: dict[str, int]
    background: pd.DataFrame
    warnings: list[str]
    mitigation: MitigationReport
    context: dict


class AutoMLService:
    def train(self, df: pd.DataFrame, config: TrainingConfig) -> TrainingOutput:
        if config.target not in df.columns:
            raise DatasetError("TARGET_NOT_FOUND", "A coluna de resultado selecionada não existe na base.")
        required = config.sensitive_features + ([config.identifier_column] if config.identifier_column else [])
        if set(required) - set(df.columns):
            raise DatasetError("INVALID_COLUMNS", "Identificador ou atributo sensível não existe na base.")
        working = df.dropna(subset=[config.target]).drop_duplicates().reset_index(drop=True)
        if config.target_rule:
            numeric = parse_numeric(working[config.target])
            working = working.loc[numeric.notna()].copy()
            numeric = numeric.loc[working.index]
            mask = numeric >= config.target_rule.value if config.target_rule.operator == "ge" else numeric <= config.target_rule.value
            y_text = pd.Series("Demais valores", index=working.index).mask(mask, config.positive_class)
        else:
            y_text = working[config.target].astype(str)
        if y_text.nunique() < 2:
            raise DatasetError("TARGET_SINGLE_CLASS", "O resultado precisa de pelo menos duas classes.")
        if config.positive_class not in set(y_text):
            raise DatasetError("INVALID_POSITIVE_CLASS", "A classe de risco selecionada não existe na base.")
        y = (y_text == config.positive_class).astype(int)
        if len(working) < 30 or y.value_counts().min() < 8:
            raise DatasetError("INSUFFICIENT_DATA", "São necessários pelo menos 30 registros e 8 exemplos de cada classe binária para treino, validação e teste.")
        trainval, test = train_test_split(working.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        train, validation = train_test_split(trainval, test_size=0.25, stratify=y.loc[trainval], random_state=RANDOM_STATE)
        # Inferências estatísticas que decidem features não consultam validação/teste.
        profile = DatasetProfiler().profile(working.loc[train], config.target)
        identifiers = set(profile.potential_ids) | ({config.identifier_column} if config.identifier_column else set())
        if config.identifier_column and working[config.identifier_column].dropna().duplicated().any():
            raise DatasetError("REPEATED_STUDENTS", "Há múltiplos registros do mesmo aluno. Use uma observação por aluno para evitar compartilhá-lo entre treino e teste.")
        sensitive = set(profile.potential_sensitive_features) | set(config.sensitive_features)
        blocked = identifiers | {config.target}
        if not config.allow_sensitive_features:
            blocked |= sensitive
        high = {a.column for a in profile.leakage_alerts if a.risk == "HIGH"} - set(config.reviewed_leakage)
        blocked |= high
        blocked |= {p.name for p in profile.column_profiles if ColumnRole.IGNORE in p.tags}
        features = list(dict.fromkeys(config.prediction_features)) if config.prediction_features is not None else [c for c in df.columns if c not in blocked]
        if set(features) - set(df.columns) or set(features) & blocked:
            raise DatasetError("UNSAFE_FEATURES", "A seleção contém coluna ausente, identificador, resultado, atributo sensível não autorizado ou leakage sem revisão.")
        if not features:
            raise DatasetError("NO_FEATURES", "Não restaram variáveis para previsão. Revise os papéis e alertas da base.")
        X_train, X_validation = working.loc[train, features], working.loc[validation, features]
        evaluator = FairnessEvaluator(config.sensitive_features, config.reference_groups).fit(working.loc[train])
        selector = ModelSelector(config.strategy, config.selection_weights)
        results = []
        fitted = {}
        failed = {}
        for algorithm in dict.fromkeys(config.algorithms):
            variants = ["baseline", "reweighing"] if config.mitigation == "reweighing" else ["baseline"]
            for variant in variants:
                key = (algorithm, variant)
                try:
                    if variant == "reweighing" and algorithm == "knn":
                        raise DatasetError("UNSUPPORTED_MITIGATION", "KNN não aceita pesos por registro; somente sua versão original foi avaliada.")
                    estimator = criar_modelo(algorithm, max(1, len(train) // 2), config.balance_classes)
                    if algorithm == "xgboost" and config.balance_classes:
                        counts = y.loc[train].value_counts()
                        estimator.set_params(scale_pos_weight=float(counts[0] / counts[1]))
                    model = build_pipeline(estimator, scale=algorithm in {"logistic", "knn"})
                    training_frame = X_train
                    if variant == "reweighing":
                        model = ReweighedClassifier(model, features, config.mitigation_features or config.sensitive_features[:1])
                        training_frame = working.loc[train]
                    if config.calibrate:
                        model = CalibratedClassifierCV(model, method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE))
                    cv = {}
                    if config.cv_folds:
                        if y.loc[train].value_counts().min() < max(config.cv_folds, 5 if config.calibrate else 2):
                            raise ValueError("Insufficient class count for nested calibration/CV")
                        scores = cross_validate(model, training_frame, y.loc[train], cv=StratifiedKFold(config.cv_folds, shuffle=True, random_state=RANDOM_STATE),
                                                scoring={"recall": "recall", "roc_auc": "roc_auc", "pr_auc": "average_precision"}, error_score="raise", n_jobs=1)
                        cv = {name[5:]: {"mean": float(value.mean()), "std": float(value.std())} for name, value in scores.items() if name.startswith("test_")}
                    model.fit(training_frame, y.loc[train])
                    metrics, prediction = evaluate_model(model, X_validation, y.loc[validation], config.threshold)
                    fairness = evaluator.evaluate(working.loc[validation], y.loc[validation], prediction)
                    score, components, weights = selector.score(metrics, fairness)
                    results.append(CandidateResult(algorithm=algorithm, variant=variant, validation=metrics, fairness=fairness, score=score,
                                                   score_components=components, effective_weights=weights, cross_validation=cv))
                    fitted[key] = model
                except Exception as exc:
                    logger.warning("Algoritmo %s/%s falhou: %s", algorithm, variant, type(exc).__name__)
                    failed[f"{algorithm}:{variant}"] = str(exc) if isinstance(exc, DatasetError) else "Treino incompatível com a base ou configuração; revise amostras, calibração e validação cruzada."
        if not results:
            raise DatasetError("ALL_MODELS_FAILED", "Nenhum algoritmo pôde ser treinado. Revise a quantidade de registros e as opções avançadas.")
        results.sort(key=lambda r: (-r.score, r.algorithm, r.variant))
        algorithm = results[0].algorithm
        model = fitted[(algorithm, results[0].variant)]
        # Não reajustar após selecionar: mantém o modelo efetivamente avaliado e teste intacto.
        metrics, predictions = evaluate_model(model, working.loc[test, features], y.loc[test], config.threshold)
        fairness = evaluator.evaluate(working.loc[test], y.loc[test], predictions)
        core = core_pipeline(model)
        importance = {} if config.calibrate else extrair_importancia(core.named_steps["model"], core.named_steps["columns"].get_feature_names_out())
        warnings = [f"{len(df) - len(working)} registros duplicados ou sem resultado removidos."]
        if not config.sensitive_features:
            warnings.append("Fairness não avaliada: nenhum atributo sensível foi confirmado.")
        if y_text.nunique() > 2:
            warnings.append("Classificação binária: classe confirmada versus todas as demais.")
        if config.balance_classes:
            warnings.append("Pesos de classe ativados em Logistic Regression, Random Forest e XGBoost; KNN não suporta essa opção.")
        if any(r.fairness.score is None for r in results):
            warnings.append("Fairness indisponível na seleção: pesos dos critérios disponíveis foram renormalizados e estão registrados.")
        from app.services.explanation_context import build_context
        return TrainingOutput(model=model, algorithm=algorithm, comparison=results, failed=failed, features=features,
            feature_schema=core.named_steps["schema"].kinds_, excluded=[c for c in df.columns if c not in features],
            classes=sorted(y_text.unique()), metrics=metrics, fairness=fairness, feature_importance=importance,
            split={"train": len(train), "validation": len(validation), "test": len(test)},
            background=X_train.sample(min(len(X_train), 40), random_state=RANDOM_STATE), warnings=warnings,
            mitigation=mitigation_report(config, results, fitted, evaluator, working.loc[test], y.loc[test], failed),
            context=build_context(X_train, y.loc[train]))
