from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def treinar_modelo(tipo, X, y):
    if tipo == "logistic":
        model = LogisticRegression(max_iter=1000)

    elif tipo == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=5,
            random_state=42
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
            n_jobs=-1
        )

    else:
        raise ValueError("Modelo inválido")

    model.fit(X, y)
    return model


def avaliar_modelo(model, X, y):
    y_pred = model.predict(X)

    try:
        y_prob = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_prob)
    except:
        roc_auc = None

    cm = confusion_matrix(y, y_pred)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }, y_pred
<<<<<<< HEAD
=======
    
def extrair_importancia(model, feature_names):
    importancias = {}

    # Logistic Regression
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        importancias = {
            feature_names[i]: float(coefs[i])
            for i in range(len(feature_names))
        }

    elif hasattr(model, "feature_importances_"):
        importancias = {
            feature_names[i]: float(model.feature_importances_[i])
            for i in range(len(feature_names))
        }

    return importancias
>>>>>>> 14ee90717e59ecdfe7b40d518c7a047aca4aded3
