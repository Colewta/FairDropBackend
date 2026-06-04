from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def treinar_modelo(tipo, X, y):
    if tipo == "logistic":
        model = LogisticRegression(max_iter=1000)

    elif tipo == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=5,
            random_state=42,
        )

    elif tipo == "knn":
        model = KNeighborsClassifier(n_neighbors=min(5, len(X)))

    elif tipo in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError(
                "XGBoost nao esta instalado. Instale a dependencia 'xgboost'."
            ) from exc

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError("Modelo invalido")

    model.fit(X, y)
    return model


def avaliar_modelo(model, X, y):
    y_pred = model.predict(X)

    try:
        y_prob = model.predict_proba(X)[:, 1]
        roc_auc = float(roc_auc_score(y, y_prob))
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }, y_pred


def extrair_importancia(model, feature_names):
    if hasattr(model, "coef_"):
        values = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return {}

    return {
        str(feature_names[index]): float(values[index])
        for index in range(len(feature_names))
    }
