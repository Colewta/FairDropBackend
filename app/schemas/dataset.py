from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ColumnRole(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    TARGET = "TARGET"
    SENSITIVE = "SENSITIVE"
    DEMOGRAPHIC = "DEMOGRAPHIC"
    ACADEMIC_PERFORMANCE = "ACADEMIC_PERFORMANCE"
    ATTENDANCE = "ATTENDANCE"
    ENGAGEMENT = "ENGAGEMENT"
    FINANCIAL = "FINANCIAL"
    TEMPORAL = "TEMPORAL"
    GENERAL_FEATURE = "GENERAL_FEATURE"
    IGNORE = "IGNORE"
    UNKNOWN = "UNKNOWN"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_percentage: float
    unique_values: int
    cardinality_ratio: float
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    role_suggestion: ColumnRole
    tags: list[ColumnRole]
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]


class Candidate(BaseModel):
    column: str
    confidence: float
    reason: str


class LeakageAlert(BaseModel):
    column: str
    risk: Literal["HIGH", "MEDIUM"]
    reason: str


class HealthIssue(BaseModel):
    severity: Literal["warning", "critical"]
    code: str
    message: str
    column: str | None = None


class DataHealth(BaseModel):
    issues: list[HealthIssue]
    checks: list[str]
    # Sem nota agregada: não mascarar um problema crítico com médias.


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    duplicates: int
    missing_cells_percentage: float
    column_profiles: list[ColumnProfile]
    target_candidates: list[Candidate]
    potential_targets: list[str]
    potential_sensitive_features: list[str]
    potential_ids: list[str]
    selected_target: str | None
    target_source: Literal["user", "suggested", "none"]
    leakage_alerts: list[LeakageAlert]
    prediction_features: list[str]
    fairness_features: list[str]
    excluded_features: list[str]
    health: DataHealth
    import_info: dict[str, str | int] = Field(default_factory=dict)
    import_warnings: list[str] = Field(default_factory=list)


class DatasetError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
