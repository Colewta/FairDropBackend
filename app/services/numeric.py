import re

import pandas as pd


def parse_numeric(series: pd.Series) -> pd.Series:
    """Aceita números regionais sem extrair dígitos de códigos alfanuméricos."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def normalize(value):
        if pd.isna(value):
            return None
        text = str(value).strip().replace("\u00a0", "")
        if not re.fullmatch(r"[+-]?(?:\d[\d.,]*|[.,]\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|Infinity)", text):
            return None
        if "," in text and "." in text:
            text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
        elif "," in text:
            text = text.replace(",", ".")
        return text

    return pd.to_numeric(series.map(normalize), errors="coerce")
