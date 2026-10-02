from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.config import RISK_THRESHOLDS
from app.schemas.dataset import DatasetProfile

Algorithm = Literal["logistic", "rf", "knn", "xgboost"]
Strategy = Literal["best_predictive", "best_fairness", "best_balanced"]


class TargetRule(BaseModel):
    operator: Literal["ge", "le"]
    value: float = Field(allow_inf_nan=False)


class TrainingConfig(BaseModel):
    mitigation: Literal["none", "reweighing"] = "none"
    mitigation_features: list[str] = Field(default_factory=list)
    dataset_id: str
    target: str
    positive_class: str
    target_rule: TargetRule | None = None
    name: str = Field(default="Modelo educacional", max_length=160)
    sensitive_features: list[str] = Field(default_factory=list)
    reference_groups: dict[str, str] = Field(default_factory=dict)
    identifier_column: str | None = None
    prediction_features: list[str] | None = None
    allow_sensitive_features: bool = False
    reviewed_leakage: list[str] = Field(default_factory=list)
    strategy: Strategy = "best_balanced"
    algorithms: list[Algorithm] = Field(default_factory=lambda: ["logistic", "rf", "knn", "xgboost"], min_length=1)
    selection_weights: dict[str, float] | None = None
    threshold: float = Field(default=0.5, gt=0, lt=1)
    low_risk_max: float = Field(default=RISK_THRESHOLDS["low"], gt=0, lt=1)
    medium_risk_max: float = Field(default=RISK_THRESHOLDS["medium"], gt=0, lt=1)
    calibrate: bool = False
    cv_folds: int = Field(default=0, ge=0, le=5)
    balance_classes: bool = True

    @model_validator(mode="after")
    def validate_options(self):
        if self.low_risk_max >= self.medium_risk_max:
            raise ValueError("As faixas de risco precisam estar em ordem crescente.")
        if self.cv_folds == 1:
            raise ValueError("Use zero ou pelo menos duas divisões de validação cruzada.")
        if self.target in self.sensitive_features or self.target == self.identifier_column:
            raise ValueError("O resultado não pode ser identificador ou atributo de fairness.")
        if set(self.mitigation_features) - set(self.sensitive_features):
            raise ValueError("Os grupos da mitigação precisam estar entre os atributos de equidade confirmados.")
        return self


class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    specificity: float
    balanced_accuracy: float
    brier_score: float
    confusion_matrix: list[list[int]]
    calibration_curve: list[dict[str, float]]


class FairnessComparison(BaseModel):
    group_rates: dict[str, float | None] = Field(default_factory=dict)
    reference_rates: dict[str, float | None] = Field(default_factory=dict)
    feature: str
    group: str
    reference: str
    group_count: int
    reference_count: int
    metrics: dict[str, float | None]
    status: str
    explanation: str


class FairnessReport(BaseModel):
    status: str
    comparisons: list[FairnessComparison] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: float | None = None
    semantics: str = "Resultado positivo representa a classe de risco confirmada; disparidade não prova discriminação."


class CandidateResult(BaseModel):
    variant: Literal["baseline", "reweighing"] = "baseline"
    algorithm: str
    validation: Metrics
    fairness: FairnessReport
    score: float
    score_components: dict[str, float]
    effective_weights: dict[str, float]
    cross_validation: dict[str, dict[str, float]] = Field(default_factory=dict)


class ExplanationFactor(BaseModel):
    interpretation: str = ""
    historical_context: str = ""
    tutor_actions: list[str] = Field(default_factory=list)
    bias_caution: str = ""
    scenario_allowed: bool = False
    reference_min: float | None = None
    reference_max: float | None = None
    feature: str
    value: str | float | None = None
    impact: float
    direction: Literal["increase", "decrease"]


class Explanation(BaseModel):
    reference_rows: int | None = None
    context_version: int = 0
    status: Literal["available", "unavailable", "pending"]
    method: str = "SHAP permutation"
    base_value: float | None = None
    probability: float | None = None
    factors: list[ExplanationFactor] = Field(default_factory=list)
    global_importance: dict[str, float] = Field(default_factory=dict)
    message: str = "Esses fatores contribuíram para a previsão do modelo; não demonstram causalidade."


class MitigationReport(BaseModel):
    method: str = "none"
    status: str = "not_requested"
    selected: bool = False
    features: list[str] = Field(default_factory=list)
    algorithm: str | None = None
    message: str = "Nenhum experimento de reponderação foi solicitado."
    before_metrics: Metrics | None = None
    after_metrics: Metrics | None = None
    before_fairness: FairnessReport | None = None
    after_fairness: FairnessReport | None = None
    weight_summary: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ModelCard(BaseModel):
    mitigation: MitigationReport = Field(default_factory=MitigationReport)
    is_demo: bool = False
    id: str
    institution_id: str
    version: str
    name: str
    created_at: str
    dataset_id: str
    dataset_name: str
    target: str
    positive_class: str
    classes: list[str]
    features: list[str]
    feature_schema: dict[str, str]
    excluded_features: list[str]
    sensitive_features: list[str]
    identifier_column: str | None
    algorithm: str
    threshold: float
    risk_thresholds: dict[str, float]
    config: TrainingConfig
    metrics: Metrics
    fairness: FairnessReport
    comparison: list[CandidateResult]
    failed_algorithms: dict[str, str]
    feature_importance: dict[str, float]
    explanation: Explanation
    split: dict[str, int]
    warnings: list[str]
    limitations: list[str]


class DatasetRecord(BaseModel):
    is_demo: bool = False
    id: str
    name: str
    created_at: str
    profile: DatasetProfile


class TrainingRun(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    model_id: str | None = None
    error: str | None = None


class StudentPrediction(BaseModel):
    groups: dict[str, str] = Field(default_factory=dict)
    id: str
    student_id: str
    batch_id: str
    model_id: str
    created_at: str
    risk_probability: float
    risk_level: Literal["low", "medium", "high"]
    prediction: int
    top_factors: list[ExplanationFactor] = Field(default_factory=list)
    explanation_status: str = "pending"


class PredictionBatch(BaseModel):
    group_filters: dict[str, list[str]] = Field(default_factory=dict)
    id: str
    model_id: str
    name: str
    created_at: str
    total: int
    summary: dict[str, int]
    average_risk: float
    warnings: list[str]
    audit: dict[str, str | float | list[str] | dict[str, float]]


class PredictionPage(BaseModel):
    total: int
    items: list[StudentPrediction]
    page: int
    page_size: int


class StudentDetail(BaseModel):
    used_information: dict[str, str | float | bool | None] = Field(default_factory=dict)
    ignored_columns: list[str] = Field(default_factory=list)
    prediction: StudentPrediction
    explanation: Explanation
    history: list[StudentPrediction]
    history_total: int


class MitigationRequest(BaseModel):
    feature: str
    strategy: Strategy = "best_balanced"


class ScenarioRequest(BaseModel):
    feature: str
    value: float = Field(allow_inf_nan=False)


class TutorReview(BaseModel):
    checked_actions: list[Literal["data_checked", "student_heard", "barriers_reviewed", "support_offered", "follow_up_planned"]] = Field(default_factory=list)
    note: str = Field(default="", max_length=2000)
    conclusion: Literal["pending", "support", "data_issue", "disagree", "monitor"] = "pending"
    updated_at: str | None = None
