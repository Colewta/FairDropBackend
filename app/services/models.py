from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier


MODEL_SPECS = {
    "logistic": {
        "nome": "Logistic Regression",
        "aliases": {"logistic", "logreg", "logistic_regression"},
    },
    "rf": {
        "nome": "Random Forest",
        "aliases": {"rf", "random_forest", "randomforest"},
    },
    "knn": {
        "nome": "KNN",
        "aliases": {"knn", "k_nearest_neighbors"},
    },
    "xgboost": {
        "nome": "XGBoost",
        "aliases": {"xgboost", "xgb"},
    },
}
MODEL_ALIASES = {
    alias: tipo
    for tipo, spec in MODEL_SPECS.items()
    for alias in spec["aliases"]
}


def listar_modelos_suportados():
    return {
        tipo: spec["nome"]
        for tipo, spec in MODEL_SPECS.items()
    }


def normalizar_tipo_modelo(tipo):
    tipo_normalizado = MODEL_ALIASES.get(str(tipo or "").strip().lower())
    if tipo_normalizado is None:
        raise ValueError("Modelo invalido")

    return tipo_normalizado


def obter_nome_modelo(tipo):
    return MODEL_SPECS[normalizar_tipo_modelo(tipo)]["nome"]


def treinar_modelo(tipo, X, y):
    tipo = normalizar_tipo_modelo(tipo)

    if tipo == "logistic":
        model = LogisticRegression(max_iter=1000)

    elif tipo == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=5,
            random_state=42,
        )

    elif tipo == "knn":
        model = KNeighborsClassifier(n_neighbors=max(1, min(5, len(X))))

    elif tipo == "xgboost":
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


def treinar_todos_modelos(X, y, tipos=None):
    tipos_modelos = tipos or MODEL_SPECS.keys()
    modelos = {}
    erros = {}

    for tipo in tipos_modelos:
        tipo_normalizado = normalizar_tipo_modelo(tipo)
        try:
            modelos[tipo_normalizado] = treinar_modelo(tipo_normalizado, X, y)
        except Exception as exc:
            erros[tipo_normalizado] = str(exc)

    return modelos, erros


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
