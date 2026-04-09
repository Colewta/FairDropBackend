import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


POSITIVE_TARGET_LABELS = {"1", "true", "yes", "sim", "positivo", "dropout"}


def carregar_dataset(caminho):
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [col.strip() for col in df.columns]
    return df


def _normalizar_string_numerica(valor):
    if pd.isna(valor):
        return valor

    texto = str(valor).strip()
    if not texto:
        return pd.NA

    if texto.count(".") > 1 and "," not in texto:
        primeira_parte, restante = texto.split(".", 1)
        texto = primeira_parte + "." + restante.replace(".", "")

    if texto.count(",") > 1 and "." not in texto:
        primeira_parte, restante = texto.split(",", 1)
        texto = primeira_parte + "." + restante.replace(",", "")
    elif texto.count(",") == 1 and "." not in texto:
        texto = texto.replace(",", ".")

    return texto


def _converter_colunas_numericas(df, target):
    df = df.copy()

    for col in df.columns:
        if col == target or pd.api.types.is_numeric_dtype(df[col]):
            continue

        serie_original = df[col].astype("string").str.strip()
        serie_normalizada = serie_original.map(_normalizar_string_numerica)
        serie_convertida = pd.to_numeric(serie_normalizada, errors="coerce")

        if serie_convertida.notna().sum() == serie_original.notna().sum():
            df[col] = serie_convertida
        else:
            df[col] = serie_original

    return df


def _binarizar_target(serie_target):
    serie = serie_target.astype("string").str.strip()
    serie_sem_nulos = serie.dropna()
    valores_unicos = [valor for valor in serie_sem_nulos.unique().tolist() if valor != ""]

    if not valores_unicos:
        raise ValueError("A coluna target nao possui valores validos.")

    if len(valores_unicos) == 2:
        mapa = {}
        for valor in valores_unicos:
            chave = valor.lower()
            mapa[valor] = 1 if chave in POSITIVE_TARGET_LABELS else 0

        if len(set(mapa.values())) == 1:
            mapa = {valores_unicos[0]: 0, valores_unicos[1]: 1}

        return serie.map(mapa).astype("int64"), {
            "target_original": valores_unicos,
            "target_binarizado": mapa
        }

    if any(valor.lower() == "dropout" for valor in valores_unicos):
        mapa = {valor: int(valor.lower() == "dropout") for valor in valores_unicos}
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

    df_preparado = df.copy()
    df_preparado = _converter_colunas_numericas(df_preparado, target)
    df_preparado = df_preparado.dropna(subset=[target])

    df_preparado[target], info_target = _binarizar_target(df_preparado[target])
    df_preparado = df_preparado.dropna().reset_index(drop=True)

    return df_preparado, {
        "linhas_apos_limpeza": len(df_preparado),
        "target_binarizado": info_target["target_binarizado"]
    }


def preprocessar(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, dummy_na=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
