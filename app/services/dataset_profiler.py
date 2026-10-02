import math
import warnings

import numpy as np
import pandas as pd

from app.core.config import IDENTIFIER_RATIO, MIN_IDENTIFIER_ROWS
from app.schemas.dataset import Candidate, ColumnProfile, ColumnRole as Role, DatasetError, DatasetProfile
from app.services.data_validator import DataValidator
from app.services.leakage_detector import LeakageDetector
from app.services.semantic_analyzer import LocalSemanticAnalyzer, SemanticAnalyzer, normalize_name
from app.services.numeric import parse_numeric

OUTCOME_LABELS = {"ativo", "evadido", "evasao", "dropout", "enrolled", "graduate", "formado", "matriculado"}
BOOLEAN_LABELS = {"true", "false", "sim", "nao", "yes", "no", "0", "1"}


def infer_type(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "unknown"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    labels = {normalize_name(str(value)) for value in values.unique()}
    if labels <= BOOLEAN_LABELS or pd.api.types.is_bool_dtype(series):
        return "boolean"
    numeric = parse_numeric(values)
    if numeric.notna().all():
        return "numeric"
    strings = values.astype(str)
    if strings.str.match(r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})").mean() >= 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dates = pd.to_datetime(strings, errors="coerce", format="mixed")
        if dates.notna().mean() >= 0.9:
            return "date"
    if values.nunique() <= 20 or values.nunique() / len(values) <= 0.2:
        return "categorical"
    return "text"


def finite_number(value: float) -> float | None:
    return float(value) if pd.notna(value) and math.isfinite(value) else None


class DatasetProfiler:
    def __init__(self, semantic: SemanticAnalyzer | None = None):
        self.semantic = semantic or LocalSemanticAnalyzer()

    def profile(self, df: pd.DataFrame, target: str | None = None) -> DatasetProfile:
        if df.empty:
            raise DatasetError("EMPTY_DATASET", "A base não contém registros.")
        if not df.columns.is_unique or not all(isinstance(c, str) and c.strip() for c in df.columns):
            raise DatasetError("INVALID_COLUMNS", "Use nomes de colunas preenchidos e únicos.")
        if target is not None and target not in df.columns:
            raise DatasetError("TARGET_NOT_FOUND", "A coluna selecionada como resultado não existe na base.")
        semantics = self.semantic.analyze_columns(list(df.columns))
        profiles = []
        candidates = []
        identifiers = []
        sensitive = []
        for column in df.columns:
            series = df[column]
            unique = int(series.nunique())
            ratio = unique / len(df)
            dtype = infer_type(series)
            tags = list(semantics[column])
            reasons = ["Correspondência do nome com o dicionário local."] if tags else []
            confidence = 0.9 if tags else 0.4
            # Cardinalidade sozinha não torna notas, datas ou medidas contínuas IDs.
            id_pattern = series.dropna().astype(str).str.fullmatch(
                r"(?:[^\s@]+@[^\s@]+\.[^\s@]+|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|[A-Za-z]{1,8}[-_]?\d{3,})"
            )
            if (len(df) >= MIN_IDENTIFIER_ROWS and ratio > IDENTIFIER_RATIO
                    and not tags and len(id_pattern) and id_pattern.mean() >= 0.9):
                tags.append(Role.IDENTIFIER)
                confidence = 0.75
                reasons.append("Mais de 95% de valores únicos com formato de identificador.")
            if dtype == "date" and Role.TEMPORAL not in tags:
                tags.append(Role.TEMPORAL)
            if Role.IDENTIFIER in tags:
                identifiers.append(column)
            labels = {normalize_name(str(v)) for v in series.dropna().unique()} if unique <= 20 else set()
            income_classes = unique == 2 and all(str(v).strip().rstrip('.').upper() in {"<=50K", ">50K"} for v in series.dropna().unique())
            if income_classes:
                tags = [tag for tag in tags if tag != Role.SENSITIVE]
                tags.append(Role.TARGET)
                reasons.append("Classes de resultado de renda reconhecidas; não representa renda contínua.")
            if Role.SENSITIVE in tags:
                sensitive.append(column)
            if (2 <= unique <= 20 and Role.IDENTIFIER not in tags and Role.SENSITIVE not in tags
                    and dtype != "date"):
                named = Role.TARGET in tags and Role.TEMPORAL not in tags
                outcome = bool(labels & OUTCOME_LABELS)
                if named or outcome or (unique == 2 and not tags):
                    canonical = normalize_name(column) in {"target", "label", "y", "class", "classe"}
                    score = 0.98 if canonical or income_classes else 0.95 if named and outcome else 0.9 if named else 0.8 if outcome else 0.45
                    reason = "Nome compatível com resultado e baixa cardinalidade." if named else (
                        "Valores compatíveis com situação acadêmica." if outcome else "Coluna binária; confirme se representa o resultado.")
                    candidates.append(Candidate(column=column, confidence=score, reason=reason))
            if unique <= 1:
                tags.append(Role.IGNORE)
                reasons.append("Coluna vazia ou constante.")
            if not tags:
                tags = [Role.GENERAL_FEATURE if dtype != "unknown" else Role.UNKNOWN]
            role = next((r for r in (Role.IGNORE, Role.IDENTIFIER, Role.SENSITIVE, Role.TARGET) if r in tags), tags[0])
            stats = {}
            if dtype == "numeric" and Role.IDENTIFIER not in tags:
                numeric = parse_numeric(series).replace([np.inf, -np.inf], np.nan)
                stats = {"min": finite_number(numeric.min()), "max": finite_number(numeric.max()),
                         "mean": finite_number(numeric.mean())}
            profiles.append(ColumnProfile(name=column, dtype=dtype, missing_percentage=round(float(series.isna().mean()) * 100, 3),
                                          unique_values=unique, cardinality_ratio=round(ratio, 6),
                                          role_suggestion=role, tags=tags, confidence=confidence,
                                          reasons=reasons or ["Tipo e cardinalidade observados; papel requer revisão."], **stats))
        candidates.sort(key=lambda c: (-c.confidence, c.column))
        selected = target if target is not None else (candidates[0].column if candidates and candidates[0].confidence >= 0.8 else None)
        leakage = LeakageDetector().detect(df, selected, identifiers)
        excluded = set(identifiers + sensitive)
        excluded.update(p.name for p in profiles if Role.IGNORE in p.tags)
        excluded.update(a.column for a in leakage if a.risk == "HIGH")
        if selected:
            excluded.add(selected)
            for profile in profiles:
                if profile.name == selected:
                    profile.role_suggestion = Role.TARGET
                    if Role.TARGET not in profile.tags:
                        profile.tags.append(Role.TARGET)
        return DatasetProfile(
            rows=len(df), columns=len(df.columns), duplicates=int(df.duplicated().sum()),
            missing_cells_percentage=round(float(df.isna().sum().sum()) / df.size * 100, 3),
            column_profiles=profiles, target_candidates=candidates,
            potential_targets=[c.column for c in candidates], potential_sensitive_features=sensitive,
            potential_ids=identifiers, selected_target=selected,
            target_source="user" if target is not None else "suggested" if selected else "none",
            leakage_alerts=leakage, prediction_features=[c for c in df.columns if c not in excluded],
            fairness_features=[c for c in sensitive if c != selected],
            excluded_features=[c for c in df.columns if c in excluded],
            health=DataValidator().check(df, selected, leakage),
            import_info=df.attrs.get("import_info", {}), import_warnings=df.attrs.get("import_warnings", []),
        )
