import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

<<<<<<< HEAD

POSITIVE_TARGET_LABELS = {
    "1", "true", "t", "yes", "y", "sim", "s", "positivo", "positive",
    "aprovado", "success", "sucesso", "dropout", "evadiu"
}
NEGATIVE_TARGET_LABELS = {
    "0", "false", "f", "no", "n", "nao", "negativo", "negative",
    "reprovado", "fail", "falha", "graduate", "graduated", "enrolled",
    "matriculado", "nao evadiu"
}
NA_VALUES = [
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "?", "-", "--",
    "sem informacao", "desconhecido"
]
NORMALIZED_NA_VALUES = set(NA_VALUES)
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1")
CSV_SEPARATORS = (None, ",", ";", "\t", "|")
NATIONALITY_COLUMNS = {"nacionality", "nationality", "nacionalidade"}
=======
POSITIVE_TARGET_LABELS = {"1", "true", "yes", "sim", "positivo", "dropout"}
>>>>>>> 14ee90717e59ecdfe7b40d518c7a047aca4aded3

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
        if valor in POSITIVE_TARGET_LABELS:
            mapa[valor] = 1
        elif valor in NEGATIVE_TARGET_LABELS:
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
    if len(valores_numericos) == 2 and serie_numerica.notna().sum() == serie_sem_nulos.size:
        menor, maior = valores_numericos
        mapa = {menor: 0, maior: 1}
        return serie_numerica.map(mapa).astype("int64"), {
            "target_original": valores_numericos,
            "target_binarizado": mapa
        }

    if len(valores_unicos) == 2:
        mapa = {}
        for valor in valores_unicos:
            chave = _chave_texto(valor)
            if chave in POSITIVE_TARGET_LABELS:
                mapa[valor] = 1
            elif chave in NEGATIVE_TARGET_LABELS:
                mapa[valor] = 0

        if len(mapa) != 2 or len(set(mapa.values())) == 1:
            mapa = {valores_unicos[0]: 0, valores_unicos[1]: 1}

        return serie.map(mapa).astype("int64"), {
            "target_original": valores_unicos,
            "target_binarizado": mapa
        }

    positivos_encontrados = [
        valor for valor in valores_unicos
        if _chave_texto(valor) in POSITIVE_TARGET_LABELS
    ]
    if len(positivos_encontrados) == 1:
        positivo = _chave_texto(positivos_encontrados[0])
        mapa = {valor: int(_chave_texto(valor) == positivo) for valor in valores_unicos}
        return serie.map(mapa).astype("int64"), {
            "target_original": valores_unicos,
            "target_binarizado": mapa
        }

    raise ValueError(
        "A coluna target precisa ser binaria ou conter a classe 'Dropout' para binarizacao automatica."
    )


def preparar_dataframe(df, target):
    if target not in df.columns:
        raise ValueError("Target invalido")

    df_preparado = _limpar_dataframe_bruto(df)
    df_preparado = _converter_colunas_numericas(df_preparado, target)
    df_preparado = _binarizar_nacionalidade(df_preparado)
    df_preparado = df_preparado.dropna(subset=[target])

    df_preparado[target], info_target = _binarizar_target(df_preparado[target])
    df_preparado = _imputar_valores_ausentes(df_preparado, target)
    df_preparado = df_preparado.reset_index(drop=True)

    contagem_classes = df_preparado[target].value_counts()
    if len(contagem_classes) < 2:
        raise ValueError("A coluna target precisa ter pelo menos duas classes validas.")
    if len(df_preparado) < 3:
        raise ValueError("Dataset insuficiente apos o preprocessamento.")

    return df_preparado, {
        "linhas_apos_limpeza": len(df_preparado),
        "target_binarizado": info_target["target_binarizado"]
    }


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
