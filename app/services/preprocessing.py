"""Transformações serializáveis, ajustadas dentro de cada split de treino."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas.dataset import DatasetError
from app.services.dataset_profiler import infer_type
from app.services.numeric import parse_numeric


class SchemaTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.kinds_ = {c: infer_type(X[c]) for c in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise DatasetError("MISSING_FEATURES", "A base não contém as colunas usadas no treino: " + ", ".join(sorted(missing)))
        frame = pd.DataFrame(index=X.index)
        for column, kind in self.kinds_.items():
            values = X[column]
            if kind == "numeric":
                frame[column] = parse_numeric(values).astype(float)
            elif kind == "date":
                dates = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
                frame[column] = (dates - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
            else:
                frame[column] = values.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
        return frame.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler() if scale else "passthrough")])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", max_categories=64, sparse_output=True))])
    columns = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=1.0)
    return Pipeline([("schema", SchemaTransformer()), ("columns", columns), ("model", estimator)])
