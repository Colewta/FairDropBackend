from typing import Literal

import numpy as np
import pandas as pd

from app.core.config import IMBALANCE_RATIO
from app.schemas.dataset import DataHealth, HealthIssue, LeakageAlert


class DataValidator:
    def check(self, df: pd.DataFrame, target: str | None,
              leakage: list[LeakageAlert]) -> DataHealth:
        issues: list[HealthIssue] = []
        checks = []

        def add(code: str, message: str, column: str | None = None,
                severity: Literal["warning", "critical"] = "warning") -> None:
            issues.append(HealthIssue(code=code, message=message, column=column, severity=severity))

        if len(df) < 30:
            add("SMALL_DATASET", "Menos de 30 registros: a avaliação pode ser instável.")
        else:
            checks.append("Pelo menos 30 registros disponíveis; isso não garante representatividade.")
        duplicates = int(df.duplicated().sum())
        if duplicates:
            add("DUPLICATES", f"{duplicates} linhas duplicadas; revise antes de separar treino e teste.")
        for column in df.columns:
            missing = float(df[column].isna().mean())
            if missing:
                add("MISSING_VALUES", f"{missing:.1%} dos valores estão ausentes.", column)
            if df[column].nunique() <= 1:
                add("CONSTANT_COLUMN", "Coluna vazia ou constante, sem variação útil para previsão.", column)
            numeric = pd.to_numeric(df[column], errors="coerce")
            if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
                add("NON_FINITE_VALUES", "Há valores infinitos; precisam ser tratados antes do treino.", column)
        if target:
            counts = df[target].value_counts()
            if len(counts) < 2:
                add("TARGET_SINGLE_CLASS", "O resultado selecionado possui menos de duas classes válidas.", target, "critical")
            else:
                checks.append("Resultado identificado com pelo menos duas classes; confirme seu significado.")
                if counts.min() < 2:
                    add("INSUFFICIENT_CLASS_ROWS", "Uma classe tem menos de dois registros; split estratificado inviável.", target, "critical")
                if counts.min() / counts.sum() < IMBALANCE_RATIO:
                    add("CLASS_IMBALANCE", f"A menor classe representa {counts.min() / counts.sum():.1%} dos registros válidos.", target)
        else:
            add("TARGET_NOT_CONFIRMED", "Nenhum resultado selecionado; confirme a coluna que deseja prever.")
        for alert in leakage:
            add("POTENTIAL_LEAKAGE", alert.reason, alert.column,
                "critical" if alert.risk == "HIGH" else "warning")
        return DataHealth(issues=issues, checks=checks)
