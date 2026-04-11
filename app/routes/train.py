from fastapi import APIRouter, UploadFile, Form, HTTPException, Body
import pandas as pd

from app.utils.file_handler import salvar_csv
from app.services.preprocess import carregar_dataset, preparar_dataframe, preprocessar
from app.services.models import treinar_modelo, avaliar_modelo, extrair_importancia
from app.services.fairness import avaliar_fairness_aif360

router = APIRouter()

GLOBAL_MODEL = None
GLOBAL_FEATURES = None

@router.post("/train")
async def train(
    file: UploadFile,
    target: str = Form(...),
    sensitive: str = Form(...),
    model_type: str = Form(...)
):
    global GLOBAL_MODEL, GLOBAL_FEATURES

    try:
        path = salvar_csv(file)

        target = target.strip()
        sensitive = sensitive.strip()

        df = carregar_dataset(path)

        if target not in df.columns:
            raise HTTPException(status_code=400, detail="Target inválido")

        if sensitive not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna sensível inválida")

        df_preparado, info_preprocessamento = preparar_dataframe(df, target)

        if sensitive not in df_preparado.columns:
            raise HTTPException(
                status_code=400,
                detail="Coluna sensível inválida após preprocessamento"
            )

        X_train, X_test, y_train, y_test = preprocessar(df_preparado, target)

        model = treinar_modelo(model_type, X_train, y_train)

        GLOBAL_MODEL = model
        GLOBAL_FEATURES = X_train.columns.tolist()

        feature_importance = extrair_importancia(model, X_train.columns)
        metricas, y_pred = avaliar_modelo(model, X_test, y_test)
        df_test = df_preparado.loc[y_test.index]

        fairness = avaliar_fairness_aif360(
            df_test,
            y_test,
            y_pred,
            target,
            sensitive
        )

        return {
            "modelo": model_type,
            "metricas": metricas,
            "fairness": fairness,
            "feature_importance": feature_importance,
            "dataset": {
                "total_linhas": len(df),
                "linhas_apos_limpeza": info_preprocessamento["linhas_apos_limpeza"],
                "treino": len(X_train),
                "teste": len(X_test)
            },
            "preprocessamento": {
                "target_binarizado": info_preprocessamento["target_binarizado"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate")
def simulate(data: dict = Body(...)):
    global GLOBAL_MODEL, GLOBAL_FEATURES

    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=400, detail="Modelo não treinado")

    try:
        missing = set(GLOBAL_FEATURES) - set(data.keys())

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Faltando features: {list(missing)}"
            )

        df = pd.DataFrame([data])
        df = df[GLOBAL_FEATURES]
        prob = GLOBAL_MODEL.predict_proba(df)[0][1]

        return {
            "probabilidade_evasao": float(prob)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))