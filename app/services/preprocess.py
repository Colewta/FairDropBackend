import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# TIPOS DE ENTRADAS COMUNS EM TABELAS PARA LIDAR COM POSSÍVEIS DIFERENÇAS DE DADOS
BOOLEAN_POSITIVE_LABELS = {
    "1", "true", "t", "yes", "y", "sim", "s", "positivo", "positive"
}
BOOLEAN_NEGATIVE_LABELS = {
    "0", "false", "f", "no", "n", "nao", "negativo", "negative"
}
TARGET_RISK_LABELS = {
    "dropout", "evasao", "evadiu", "evadido", "abandonou", "desistente",
    "churn", "default", "inadimplente", "fraude", "fraud", "risco", "risk"
}
TARGET_FAVORABLE_LABELS = {
    "aprovado", "approved", "success", "sucesso", "graduate", "graduated",
    "formado", "concluido", "pass", "passed", "admitido", "admitted",
    "hired", "contratado", "paid", "quitado"
}
TARGET_NEGATIVE_LABELS = {
    "reprovado", "rejected", "fail", "failed", "falha", "enrolled",
    "matriculado", "suspended", "suspenso", "pending", "pendente",
    "active", "ativo", "nao evadiu", "nao aprovado"
}
TARGET_POSITIVE_GROUPS = (
    ("classe_de_risco_reconhecida", TARGET_RISK_LABELS),
    ("classe_positiva_reconhecida", BOOLEAN_POSITIVE_LABELS | TARGET_FAVORABLE_LABELS),
)
SENSITIVE_PRIVILEGED_LABELS = {
    "1", "true", "t", "yes", "y", "sim", "s", "male", "masculino",
    "homem", "m", "white", "branco", "privileged", "privilegiado"
}
SENSITIVE_UNPRIVILEGED_LABELS = {
    "0", "false", "f", "no", "n", "nao", "female", "feminino",
    "mulher", "black", "preto", "negro", "pardo", "unprivileged",
    "nao privilegiado"
}
NA_VALUES = [
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "?", "-", "--",
    "sem informacao", "desconhecido"
]
NORMALIZED_NA_VALUES = set(NA_VALUES)
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1")
CSV_SEPARATORS = (None, ",", ";", "\t", "|")
NATIONALITY_COLUMNS = {"nacionality", "nationality", "nacionalidade"}
MAX_CATEGORICAL_TARGET_CLASSES = 20
CONTINUOUS_TARGET_UNIQUE_RATIO = 0.35

# 
def carregar_dataset(caminho):
    ultimo_erro = None

    for encoding in CSV_ENCODINGS:
        for sep in CSV_SEPARATORS:
            try:
                df = pd.read_csv(
                    caminho,
                    sep=sep,
                    engine="python",
                    encoding=encoding,
                    encoding_errors="replace",
                    on_bad_lines="skip",
                    skipinitialspace=True,
                    na_values=NA_VALUES,
                    keep_default_na=True
                )
                df = _limpar_dataframe_bruto(df)

                # VERIFICA SE NÃO É UM ARQUIVO VAZIO
                if not df.empty and len(df.columns) > 1:
                    return df

                ultimo_erro = ValueError("CSV vazio ou com apenas uma coluna reconhecida.")
            except (UnicodeDecodeError, EmptyDataError, ParserError, ValueError) as erro:
                ultimo_erro = erro

    raise ValueError(f"Nao foi possivel ler o CSV enviado: {ultimo_erro}")


def _limpar_nome_coluna(coluna):
    nome = str(coluna).replace("\ufeff", "").strip()
    nome = re.sub(r"\s+", " ", nome)
    return nome


def _normalizar_colunas(colunas):
    nomes = []
    contagem = {}

    for indice, coluna in enumerate(colunas, start=1):
        nome = _limpar_nome_coluna(coluna) or f"coluna_{indice}"
        nome_base = nome
        contagem[nome_base] = contagem.get(nome_base, 0) + 1

        if contagem[nome_base] > 1:
            nome = f"{nome_base}_{contagem[nome_base]}"

        nomes.append(nome)

    return nomes


def _chave_texto(valor):
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    texto = re.sub(r"\s+", " ", texto)

    return texto


def _limpar_dataframe_bruto(df):
    df = df.copy()
    df.columns = _normalizar_colunas(df.columns)

    colunas_vazias = [
        col for col in df.columns
        if col.lower().startswith("unnamed:") and df[col].isna().all()
    ]
    if colunas_vazias:
        df = df.drop(columns=colunas_vazias)

    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    return df


def _normalizar_string_numerica(valor):
    if pd.isna(valor):
        return valor

    texto = str(valor).strip()
    if not texto:
        return pd.NA

    negativo = texto.startswith("(") and texto.endswith(")")
    texto = texto.strip("()")
    texto = re.sub(r"[^\d,.\-+]", "", texto)

    if texto.count("-") > 1 or texto.count("+") > 1:
        return pd.NA

    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif texto.count(",") > 1:
        partes = texto.split(",")
        texto = "".join(partes[:-1]) + "." + partes[-1]
    elif texto.count(".") > 1:
        partes = texto.split(".")
        texto = "".join(partes[:-1]) + "." + partes[-1]
    elif texto.count(",") == 1:
        texto = texto.replace(",", ".")

    if negativo and texto:
        texto = f"-{texto.lstrip('+-')}"

    return texto


def _normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()
    texto = re.sub(r"\s+", " ", texto)

    if not texto or _chave_texto(texto) in NORMALIZED_NA_VALUES:
        return pd.NA

    return texto


def _converter_booleanos(serie):
    serie_texto = serie.map(lambda valor: _chave_texto(valor) if not pd.isna(valor) else pd.NA)
    mapa = {}

    for valor in serie_texto.dropna().unique():
        if valor in BOOLEAN_POSITIVE_LABELS:
            mapa[valor] = 1
        elif valor in BOOLEAN_NEGATIVE_LABELS:
            mapa[valor] = 0

    if not mapa:
        return None

    convertido = serie_texto.map(mapa)
    taxa_convertida = convertido.notna().sum() / max(serie_texto.notna().sum(), 1)

    return convertido.astype("float64") if taxa_convertida >= 0.9 else None


def _converter_datas(serie):
    serie_texto = serie.astype("string").str.strip()
    candidatos_data = serie_texto.str.contains(r"[/\-.]", regex=True, na=False).sum()

    if candidatos_data / max(serie_texto.notna().sum(), 1) < 0.6:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        convertido = pd.to_datetime(serie_texto, errors="coerce", dayfirst=True)
    taxa_convertida = convertido.notna().sum() / max(serie_texto.notna().sum(), 1)

    if taxa_convertida < 0.8:
        return None

    convertido_numerico = convertido.view("int64").astype("float64") / 1_000_000_000
    convertido_numerico[convertido.isna()] = np.nan

    return convertido_numerico


def _converter_colunas_numericas(df, target):
    df = df.copy()

    for col in df.columns:
        if col == target:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue

        serie_original = df[col].map(_normalizar_texto)

        serie_bool = _converter_booleanos(serie_original)
        if serie_bool is not None:
            df[col] = serie_bool
            continue

        serie_normalizada = serie_original.map(_normalizar_string_numerica)
        serie_convertida = pd.to_numeric(serie_normalizada, errors="coerce")
        taxa_convertida = serie_convertida.notna().sum() / max(serie_original.notna().sum(), 1)

        if taxa_convertida >= 0.85:
            df[col] = serie_convertida
        else:
            serie_data = _converter_datas(serie_original)
            df[col] = serie_data if serie_data is not None else serie_original

    return df


def _binarizar_nacionalidade(df):
    df = df.copy()

    for col in df.columns:
        if _chave_texto(col) not in NATIONALITY_COLUMNS:
            continue

        nacionalidade = pd.to_numeric(
            df[col].map(_normalizar_string_numerica),
            errors="coerce"
        )
        df[col] = (nacionalidade == 1).astype("int64")

    return df


def _formatar_valor_info(valor):
    if isinstance(valor, (np.integer, int)):
        return str(int(valor))
    if isinstance(valor, (np.floating, float)):
        numero = float(valor)
        return str(int(numero)) if numero.is_integer() else str(numero)

    return str(valor)


def _montar_info_binarizacao(valores, mapa, estrategia, classe_positiva=None):
    info = {
        "classes_originais": [_formatar_valor_info(valor) for valor in valores],
        "target_binarizado": {
            _formatar_valor_info(valor): int(classe)
            for valor, classe in mapa.items()
        },
        "estrategia": estrategia,
    }

    if classe_positiva is not None:
        info["classe_positiva"] = _formatar_valor_info(classe_positiva)

    return info


def _classe_mais_frequente(valores, contagens):
    return sorted(
        valores,
        key=lambda valor: (-contagens.get(valor, 0), _chave_texto(valor))
    )[0]


def _selecionar_por_grupos_texto(valores, contagens, grupos):
    for estrategia, rotulos in grupos:
        candidatos = [
            valor for valor in valores
            if _chave_texto(valor) in rotulos
        ]

        if candidatos:
            return _classe_mais_frequente(candidatos, contagens), estrategia

    return None, None


def _selecionar_classe_positiva_texto(valores, contagens):
    positivo, estrategia = _selecionar_por_grupos_texto(
        valores,
        contagens,
        TARGET_POSITIVE_GROUPS,
    )
    if positivo is not None:
        return positivo, estrategia

    negativos = [
        valor for valor in valores
        if _chave_texto(valor) in (BOOLEAN_NEGATIVE_LABELS | TARGET_NEGATIVE_LABELS)
    ]
    if len(valores) == 2 and len(negativos) == 1:
        negativo = negativos[0]
        return next(valor for valor in valores if valor != negativo), "binario_classe_negativa_reconhecida"

    if len(valores) == 2:
        return sorted(valores, key=_chave_texto)[-1], "binario_texto_ordenado"

    elegiveis = [valor for valor in valores if contagens.get(valor, 0) >= 2]
    if not elegiveis:
        elegiveis = valores

    return _classe_mais_frequente(elegiveis, contagens), "multiclasse_classe_mais_frequente"


def _parece_target_continuo(serie_numerica):
    valores = serie_numerica.dropna().astype("float64")
    total = len(valores)

    if total == 0:
        return False

    quantidade_unicos = valores.nunique()
    if quantidade_unicos <= 2:
        return False

    taxa_unicos = quantidade_unicos / max(total, 1)
    fracao_decimais = (~np.isclose(valores, np.round(valores))).mean()

    return (
        quantidade_unicos > MAX_CATEGORICAL_TARGET_CLASSES
        and taxa_unicos >= CONTINUOUS_TARGET_UNIQUE_RATIO
    ) or (quantidade_unicos > 10 and fracao_decimais > 0.2)


def _binarizar_target_numerico(serie_numerica, valores_numericos):
    if len(valores_numericos) == 2:
        menor, maior = valores_numericos
        mapa = {menor: 0, maior: 1}
        return serie_numerica.map(mapa), _montar_info_binarizacao(
            valores_numericos,
            mapa,
            "binario_numerico",
            maior,
        )

    if _parece_target_continuo(serie_numerica):
        raise ValueError(
            "A coluna target parece continua. Este fluxo usa classificacao binaria "
            "para as metricas de fairness; escolha uma coluna categorica ou envie "
            "o target ja agrupado em classes."
        )

    classe_positiva = max(valores_numericos)
    mapa = {
        valor: int(valor == classe_positiva)
        for valor in valores_numericos
    }
    return serie_numerica.map(mapa), _montar_info_binarizacao(
        valores_numericos,
        mapa,
        "multiclasse_numerico_maior_valor_vs_resto",
        classe_positiva,
    )


def _binarizar_target_texto(serie, valores_unicos):
    contagens = serie.dropna().value_counts().to_dict()
    classe_positiva, estrategia = _selecionar_classe_positiva_texto(
        valores_unicos,
        contagens,
    )
    chave_positiva = _chave_texto(classe_positiva)
    mapa = {
        valor: int(_chave_texto(valor) == chave_positiva)
        for valor in valores_unicos
    }

    return serie.map(mapa), _montar_info_binarizacao(
        valores_unicos,
        mapa,
        estrategia,
        classe_positiva,
    )


def _binarizar_target(serie_target):
    serie = serie_target.map(_normalizar_texto)
    serie_sem_nulos = serie.dropna()
    valores_unicos = [valor for valor in serie_sem_nulos.unique().tolist() if valor != ""]

    if not valores_unicos:
        raise ValueError("A coluna target nao possui valores validos.")

    serie_numerica = pd.to_numeric(
        serie.map(_normalizar_string_numerica),
        errors="coerce"
    )
    valores_numericos = sorted(serie_numerica.dropna().unique().tolist())
    target_totalmente_numerico = serie_numerica.notna().sum() == serie_sem_nulos.size

    if target_totalmente_numerico:
        return _binarizar_target_numerico(serie_numerica, valores_numericos)

    return _binarizar_target_texto(serie, valores_unicos)


def _selecionar_grupo_privilegiado_texto(valores, contagens):
    grupos = (
        ("sensivel_grupo_privilegiado_reconhecido", SENSITIVE_PRIVILEGED_LABELS),
    )
    privilegiado, estrategia = _selecionar_por_grupos_texto(valores, contagens, grupos)
    if privilegiado is not None:
        return privilegiado, estrategia

    nao_privilegiados = [
        valor for valor in valores
        if _chave_texto(valor) in SENSITIVE_UNPRIVILEGED_LABELS
    ]
    if len(valores) == 2 and len(nao_privilegiados) == 1:
        nao_privilegiado = nao_privilegiados[0]
        return next(valor for valor in valores if valor != nao_privilegiado), "sensivel_grupo_nao_privilegiado_reconhecido"

    if len(valores) == 2:
        return sorted(valores, key=_chave_texto)[-1], "sensivel_binario_texto_ordenado"

    return _classe_mais_frequente(valores, contagens), "sensivel_multiclasse_grupo_mais_frequente_vs_resto"


def _binarizar_coluna_sensivel(serie_sensitive):
    serie = serie_sensitive.map(_normalizar_texto)
    serie_sem_nulos = serie.dropna()
    valores_unicos = [valor for valor in serie_sem_nulos.unique().tolist() if valor != ""]

    if len(valores_unicos) < 2:
        raise ValueError("A coluna sensivel precisa ter pelo menos dois grupos validos.")

    serie_numerica = pd.to_numeric(
        serie.map(_normalizar_string_numerica),
        errors="coerce"
    )
    valores_numericos = sorted(serie_numerica.dropna().unique().tolist())

    if serie_numerica.notna().sum() == serie_sem_nulos.size:
        if len(valores_numericos) == 2:
            menor, maior = valores_numericos
            mapa = {menor: 0, maior: 1}
            return serie_numerica.map(mapa), {
                "classes_originais": [_formatar_valor_info(valor) for valor in valores_numericos],
                "sensitive_binarizado": {
                    _formatar_valor_info(valor): int(classe)
                    for valor, classe in mapa.items()
                },
                "grupo_privilegiado": _formatar_valor_info(maior),
                "estrategia": "sensivel_binario_numerico",
            }

        contagens_numericas = serie_numerica.dropna().value_counts().to_dict()
        grupo_privilegiado = sorted(
            valores_numericos,
            key=lambda valor: (-contagens_numericas.get(valor, 0), valor)
        )[0]
        mapa = {
            valor: int(valor == grupo_privilegiado)
            for valor in valores_numericos
        }
        return serie_numerica.map(mapa), {
            "classes_originais": [_formatar_valor_info(valor) for valor in valores_numericos],
            "sensitive_binarizado": {
                _formatar_valor_info(valor): int(classe)
                for valor, classe in mapa.items()
            },
            "grupo_privilegiado": _formatar_valor_info(grupo_privilegiado),
            "estrategia": "sensivel_multiclasse_grupo_mais_frequente_vs_resto",
        }

    contagens = serie_sem_nulos.value_counts().to_dict()
    grupo_privilegiado, estrategia = _selecionar_grupo_privilegiado_texto(
        valores_unicos,
        contagens,
    )
    chave_privilegiada = _chave_texto(grupo_privilegiado)
    mapa = {
        valor: int(_chave_texto(valor) == chave_privilegiada)
        for valor in valores_unicos
    }

    return serie.map(mapa), {
        "classes_originais": [_formatar_valor_info(valor) for valor in valores_unicos],
        "sensitive_binarizado": {
            _formatar_valor_info(valor): int(classe)
            for valor, classe in mapa.items()
        },
        "grupo_privilegiado": _formatar_valor_info(grupo_privilegiado),
        "estrategia": estrategia,
    }


def preparar_dataframe(df, target, sensitive=None):
    if target not in df.columns:
        raise ValueError("Target invalido")
    if sensitive is not None:
        if sensitive == target:
            raise ValueError("Target e coluna sensivel precisam ser diferentes.")
        if sensitive not in df.columns:
            raise ValueError("Coluna sensivel invalida")

    df_preparado = _limpar_dataframe_bruto(df)
    df_preparado = _converter_colunas_numericas(df_preparado, target)
    df_preparado = _binarizar_nacionalidade(df_preparado)
    df_preparado = df_preparado.dropna(subset=[target])

    df_preparado[target], info_target = _binarizar_target(df_preparado[target])
    df_preparado = df_preparado.dropna(subset=[target])
    df_preparado[target] = df_preparado[target].astype("int64")

    info_sensitive = None
    if sensitive is not None:
        df_preparado[sensitive], info_sensitive = _binarizar_coluna_sensivel(
            df_preparado[sensitive]
        )
        df_preparado = df_preparado.dropna(subset=[sensitive])
        df_preparado[sensitive] = df_preparado[sensitive].astype("int64")

    df_preparado = _imputar_valores_ausentes(df_preparado, target)
    df_preparado = df_preparado.reset_index(drop=True)

    contagem_classes = df_preparado[target].value_counts()
    if len(contagem_classes) < 2:
        raise ValueError("A coluna target precisa ter pelo menos duas classes validas.")
    if contagem_classes.min() < 2:
        raise ValueError(
            "Apos a binarizacao, cada classe do target precisa ter pelo menos "
            "dois registros para separar treino e teste com seguranca."
        )
    if len(df_preparado) < 3:
        raise ValueError("Dataset insuficiente apos o preprocessamento.")

    info_preprocessamento = {
        "linhas_apos_limpeza": len(df_preparado),
        "target_binarizado": info_target["target_binarizado"],
        "target_classe_positiva": info_target.get("classe_positiva"),
        "target_estrategia": info_target.get("estrategia"),
    }

    if info_sensitive is not None:
        info_preprocessamento.update({
            "sensitive_binarizado": info_sensitive["sensitive_binarizado"],
            "sensitive_grupo_privilegiado": info_sensitive["grupo_privilegiado"],
            "sensitive_estrategia": info_sensitive["estrategia"],
        })

    return df_preparado, info_preprocessamento


def _imputar_valores_ausentes(df, target):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if col == target:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            mediana = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(mediana) else mediana)
        else:
            moda = df[col].dropna().mode()
            preenchimento = moda.iloc[0] if not moda.empty else "desconhecido"
            df[col] = df[col].fillna(preenchimento).astype(str)

    return df


def preprocessar(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    contagem_classes = y.value_counts()
    estratificar = y if contagem_classes.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=estratificar
    )

    scaler = StandardScaler()
    colunas = X_train.columns

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=colunas,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=colunas,
        index=X_test.index
    )

    return X_train, X_test, y_train, y_test
