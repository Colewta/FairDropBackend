from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
import pandas as pd

from app.services.fairness import avaliar_fairness_aif360
from app.services.models import (
    avaliar_modelo,
    extrair_importancia,
    listar_modelos_suportados,
    normalizar_tipo_modelo,
    obter_nome_modelo,
    treinar_todos_modelos,
)
from app.services.preprocess import (
    analisar_dataset,
    carregar_dataset,
    preparar_dataframe,
    preprocessar,
)
from app.utils.file_handler import salvar_csv

router = APIRouter()

GLOBAL_MODELS = {}
GLOBAL_MODEL = None
GLOBAL_FEATURES = None
GLOBAL_PRIMARY_MODEL_TYPE = None


def _obter_valor_caminho(item, caminho):
    valor = item
    for chave in caminho:
        if not isinstance(valor, dict):
            return None
        valor = valor.get(chave)

    return valor


def _selecionar_melhor_modelo(resultados, caminho):
    candidatos = []

    for tipo, info in resultados.items():
        valor = _obter_valor_caminho(info, caminho)
        if valor is None:
            continue
        candidatos.append((tipo, float(valor), info))

    if not candidatos:
        return None

    tipo, valor, info = max(
        candidatos,
        key=lambda item: (item[1], item[2]["metricas"].get("accuracy", 0.0)),
    )
    return {
        "tipo": tipo,
        "nome": info["nome"],
        "valor": round(valor, 4),
    }


def _selecionar_modelo_equilibrado(resultados):
    candidatos = []

    for tipo, info in resultados.items():
        acuracia = info["metricas"].get("accuracy")
        fairness_score = info["fairness"].get("fairness_score")
        if acuracia is None:
            continue

        score_equilibrado = float(acuracia) if fairness_score is None else (
            float(acuracia) + float(fairness_score)
        ) / 2
        candidatos.append((tipo, score_equilibrado, info))

    if not candidatos:
        return None

    tipo, valor, info = max(candidatos, key=lambda item: item[1])
    return {
        "tipo": tipo,
        "nome": info["nome"],
        "valor": round(valor, 4),
    }


def _montar_comparativo_modelos(resultados, erros):
    melhor_acuracia = _selecionar_melhor_modelo(resultados, ("metricas", "accuracy"))
    melhor_fairness = _selecionar_melhor_modelo(resultados, ("fairness", "fairness_score"))
    melhor_equilibrio = _selecionar_modelo_equilibrado(resultados)

    insights = []
    if melhor_acuracia and melhor_fairness:
        if melhor_acuracia["tipo"] == melhor_fairness["tipo"]:
            insights.append(
                f"{melhor_acuracia['nome']} liderou em acuracia e fairness nesta execucao."
            )
        else:
            insights.append(
                f"{melhor_acuracia['nome']} teve a maior acuracia ({melhor_acuracia['valor']})."
            )
            insights.append(
                f"{melhor_fairness['nome']} apresentou o melhor fairness_score ({melhor_fairness['valor']})."
            )
    elif melhor_acuracia:
        insights.append(
            f"{melhor_acuracia['nome']} teve a maior acuracia ({melhor_acuracia['valor']})."
        )

    if melhor_equilibrio is not None:
        insights.append(
            f"{melhor_equilibrio['nome']} ficou com o melhor equilibrio geral entre performance e fairness."
        )

    return {
        "melhor_acuracia": melhor_acuracia,
        "melhor_fairness": melhor_fairness,
        "melhor_equilibrio": melhor_equilibrio,
        "modelos_com_falha": {
            tipo: {
                "nome": obter_nome_modelo(tipo),
                "erro": erro,
            }
            for tipo, erro in erros.items()
        },
        "insights": insights,
    }


def _escolher_modelo_principal(resultados, modelo_solicitado, comparativo):
    if modelo_solicitado and modelo_solicitado in resultados:
        return modelo_solicitado, "modelo_solicitado"

    if comparativo.get("melhor_equilibrio") is not None:
        return comparativo["melhor_equilibrio"]["tipo"], "melhor_equilibrio"

    if comparativo.get("melhor_acuracia") is not None:
        return comparativo["melhor_acuracia"]["tipo"], "melhor_acuracia"

    return next(iter(resultados)), "primeiro_modelo_disponivel"


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        path = salvar_csv(file)
        df, info_carga = carregar_dataset(path, return_metadata=True)
        analise = analisar_dataset(df, info_carga)

        return {
            "arquivo": file.filename,
            "modelos_disponiveis": listar_modelos_suportados(),
            "analise_dataset": analise,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/train")
async def train(
    file: UploadFile = File(...),
    target: str = Form(...),
    sensitive: str = Form(...),
    model_type: str = Form(""),
):
    global GLOBAL_MODELS, GLOBAL_MODEL, GLOBAL_FEATURES, GLOBAL_PRIMARY_MODEL_TYPE

    try:
        path = salvar_csv(file)

        target = target.strip()
        sensitive = sensitive.strip()
        model_type = model_type.strip()

        df, info_carga = carregar_dataset(path, return_metadata=True)
        analise = analisar_dataset(df, info_carga)

        if target not in df.columns:
            raise HTTPException(status_code=400, detail="Target invalido")

        if sensitive not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna sensivel invalida")

        df_preparado, info_preprocessamento = preparar_dataframe(df, target, sensitive)

        if sensitive not in df_preparado.columns:
            raise HTTPException(
                status_code=400,
                detail="Coluna sensivel invalida apos preprocessamento",
            )

        X_train, X_test, y_train, y_test, info_features = preprocessar(df_preparado, target)
        modelos_treinados, erros_treinamento = treinar_todos_modelos(X_train, y_train)

        if not modelos_treinados:
            raise HTTPException(
                status_code=500,
                detail={
                    "mensagem": "Nenhum modelo conseguiu ser treinado.",
                    "erros": erros_treinamento,
                },
            )

        resultados_modelos = {}
        df_test = df_preparado.loc[y_test.index].copy()

        for tipo_modelo, modelo in modelos_treinados.items():
            metricas, y_pred = avaliar_modelo(modelo, X_test, y_test)
            fairness = avaliar_fairness_aif360(
                df_test,
                y_test,
                y_pred,
                target,
                sensitive,
            )

            resultados_modelos[tipo_modelo] = {
                "nome": obter_nome_modelo(tipo_modelo),
                "metricas": metricas,
                "fairness": fairness,
                "feature_importance": extrair_importancia(modelo, X_train.columns),
            }

        comparativo = _montar_comparativo_modelos(resultados_modelos, erros_treinamento)

        modelo_solicitado = None
        if model_type:
            try:
                modelo_solicitado = normalizar_tipo_modelo(model_type)
            except ValueError:
                modelo_solicitado = None

        modelo_principal, criterio_modelo_principal = _escolher_modelo_principal(
            resultados_modelos,
            modelo_solicitado,
            comparativo,
        )
        resultado_principal = resultados_modelos[modelo_principal]

        GLOBAL_MODELS = modelos_treinados
        GLOBAL_MODEL = modelos_treinados[modelo_principal]
        GLOBAL_FEATURES = info_features["colunas_modelo"]
        GLOBAL_PRIMARY_MODEL_TYPE = modelo_principal

        return {
            "modelo": modelo_principal,
            "modelo_nome": resultado_principal["nome"],
            "modelo_solicitado": model_type or None,
            "modelo_principal": {
                "tipo": modelo_principal,
                "nome": resultado_principal["nome"],
                "criterio": criterio_modelo_principal,
            },
            "metricas": resultado_principal["metricas"],
            "fairness": resultado_principal["fairness"],
            "feature_importance": resultado_principal["feature_importance"],
            "modelos": resultados_modelos,
            "comparativo_modelos": comparativo,
            "dataset": {
                "total_linhas": analise["resumo"]["registros_encontrados"],
                "total_colunas": analise["resumo"]["colunas_encontradas"],
                "linhas_apos_limpeza": info_preprocessamento["linhas_apos_limpeza"],
                "linhas_finais": info_preprocessamento["linhas_finais"],
                "treino": len(X_train),
                "teste": len(X_test),
                "features_originais": info_features["features_originais"],
                "features_modelo": info_features["features_apos_encoding"],
            },
            "analise_dataset": analise,
            "preprocessamento": {
                "target_binarizado": info_preprocessamento["target_binarizado"],
                "target_classe_positiva": info_preprocessamento.get("target_classe_positiva"),
                "target_estrategia": info_preprocessamento.get("target_estrategia"),
                "sensitive_binarizado": info_preprocessamento.get("sensitive_binarizado"),
                "sensitive_grupo_privilegiado": info_preprocessamento.get("sensitive_grupo_privilegiado"),
                "sensitive_estrategia": info_preprocessamento.get("sensitive_estrategia"),
                "linhas_descartadas_target_nulo": info_preprocessamento["linhas_descartadas_target_nulo"],
                "linhas_descartadas_target_invalido": info_preprocessamento["linhas_descartadas_target_invalido"],
                "linhas_descartadas_sensitive_nulo": info_preprocessamento["linhas_descartadas_sensitive_nulo"],
                "valores_ausentes_preenchidos": info_preprocessamento["valores_ausentes_preenchidos"],
                "distribuicao_target": info_preprocessamento["distribuicao_target"],
            },
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate")
def simulate(data: dict = Body(...)):
    global GLOBAL_MODELS, GLOBAL_MODEL, GLOBAL_FEATURES, GLOBAL_PRIMARY_MODEL_TYPE

    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=400, detail="Modelo nao treinado")

    try:
        payload = dict(data)
        modelo_solicitado = payload.pop("model_type", None)

        modelo = GLOBAL_MODEL
        tipo_modelo = GLOBAL_PRIMARY_MODEL_TYPE

        if modelo_solicitado:
            tipo_modelo = normalizar_tipo_modelo(modelo_solicitado)
            if tipo_modelo not in GLOBAL_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modelo {tipo_modelo} nao esta disponivel para simulacao",
                )
            modelo = GLOBAL_MODELS[tipo_modelo]

        missing = set(GLOBAL_FEATURES) - set(payload.keys())

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Faltando features: {list(missing)}",
            )

        df = pd.DataFrame([payload])
        df = df[GLOBAL_FEATURES]
        prob = modelo.predict_proba(df)[0][1]

        return {
            "modelo": tipo_modelo,
            "modelo_nome": obter_nome_modelo(tipo_modelo),
            "probabilidade_evasao": float(prob),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
