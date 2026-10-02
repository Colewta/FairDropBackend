"""Contexto agregado do treino e orientações, sem inventar causas individuais."""
import pandas as pd

from app.schemas.dataset import ColumnRole
from app.services.numeric import parse_numeric
from app.services.semantic_analyzer import LocalSemanticAnalyzer

CONTEXT_VERSION = 1
MIN_CONTEXT_GROUP = 10


def _rate(y, mask):
    count = int(mask.sum())
    return {"count": count, "rate": float(y.loc[mask].mean()) if y is not None and count >= MIN_CONTEXT_GROUP else None}


def build_context(frame: pd.DataFrame, y: pd.Series | None = None) -> dict:
    context = {"source": "training" if y is not None else "reference_sample", "rows": len(frame), "columns": {}}
    for column in frame:
        series = frame[column]
        numeric = parse_numeric(series)
        item = {"count": int(series.notna().sum()), "missing": int(series.isna().sum())}
        if series.notna().any() and numeric[series.notna()].notna().all():
            clean = numeric.dropna()
            cuts = sorted(set(float(v) for v in clean.quantile([.25, .75])))
            boundaries = [None, *cuts, None]
            bins = []
            for lower, upper in zip(boundaries[:-1], boundaries[1:]):
                mask = numeric.notna()
                if lower is not None:
                    mask &= numeric > lower
                if upper is not None:
                    mask &= numeric <= upper
                bins.append({"lower": lower, "upper": upper, **_rate(y, mask), "others": _rate(y, numeric.notna() & ~mask)})
            item.update(kind="numeric", median=float(clean.median()), min=float(clean.min()), max=float(clean.max()), bins=bins)
        else:
            text = series.astype("string")
            categories = {str(value): {**_rate(y, (text == value).fillna(False)), "others": _rate(y, (text != value).fillna(False))}
                          for value in text.value_counts().head(30).index}
            item.update(kind="categorical", categories=categories)
        context["columns"][column] = item
    return context


def scenario_allowed(feature: str) -> bool:
    tags = LocalSemanticAnalyzer().analyze_columns([feature])[feature]
    return ColumnRole.SENSITIVE not in tags and ColumnRole.IDENTIFIER not in tags and bool(
        set(tags) & {ColumnRole.ATTENDANCE, ColumnRole.ACADEMIC_PERFORMANCE, ColumnRole.ENGAGEMENT})


def tutor_guidance(feature: str) -> tuple[list[str], str]:
    tags = LocalSemanticAnalyzer().analyze_columns([feature])[feature]
    if ColumnRole.SENSITIVE in tags:
        return (["Revise com a equipe técnica por que este atributo entrou no modelo.",
                 "Garanta acesso igual ao apoio; não peça ao aluno que altere uma característica pessoal."],
                "Uma característica de grupo não explica a conduta do aluno. Compare os erros entre grupos e avalie retirar proxies ou reponderar o treino.")
    if ColumnRole.ATTENDANCE in tags:
        actions = ["Confira se faltas, presenças e justificativas foram registradas corretamente.",
                   "Pergunte ao aluno se há barreiras de horário, transporte ou acessibilidade; não presuma o motivo.",
                   "Combine apoio ou alternativas institucionais disponíveis e acompanhe a frequência em outro período."]
        caution = "Registros de presença podem refletir barreiras de acesso e práticas diferentes entre turmas. Uma falta não demonstra desinteresse."
    elif ColumnRole.ACADEMIC_PERFORMANCE in tags:
        actions = ["Confira notas, recuperações e critérios de avaliação antes de interpretar o alerta.",
                   "Investigue dificuldades relatadas pelo aluno e ofereça tutoria ou recuperação acessível.",
                   "Compare critérios e acesso ao apoio entre cursos e grupos; registre o acompanhamento."]
        caution = "Notas podem refletir diferenças de avaliação e de oportunidades. Não use a previsão para reduzir oportunidades do aluno."
    elif ColumnRole.ENGAGEMENT in tags:
        actions = ["Verifique se a plataforma registra todas as formas de participação.",
                   "Pergunte sobre condições de acesso e ofereça canais alternativos quando disponíveis.",
                   "Reavalie depois do apoio, sem equiparar poucos acessos à falta de interesse."]
        caution = "Atividade digital pode funcionar como proxy de acesso à internet e equipamentos. Revise essa hipótese nos grupos afetados."
    elif ColumnRole.FINANCIAL in tags:
        actions = ["Confira a atualidade e a finalidade deste registro.",
                   "Apresente, de forma reservada e sem condicionamento, os programas institucionais de apoio disponíveis."]
        caution = "Condições financeiras não definem capacidade acadêmica. Investigue desigualdade de acesso e possíveis proxies socioeconômicos."
    else:
        actions = ["Confirme o significado, a unidade e a qualidade desta coluna com a área responsável.",
                   "Converse com o aluno antes de decidir um encaminhamento.",
                   "Verifique se o campo representa diferenças de oportunidade ou registro entre grupos."]
        caution = "O nome da coluna não permite determinar uma causa. A associação pode refletir outras variáveis ou desigualdades históricas."
    return actions, caution


def _number(value):
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def enrich_explanation(explanation, context: dict):
    if explanation.status != "available":
        return explanation
    for factor in explanation.factors:
        item = context.get("columns", {}).get(factor.feature, {})
        actions, caution = tutor_guidance(factor.feature)
        factor.tutor_actions, factor.bias_caution = actions, caution
        direction = "elevou" if factor.impact >= 0 else "reduziu"
        factor.interpretation = (f"Neste caso, o modelo atribuiu a esta informação uma contribuição que {direction} a previsão em "
                                 f"{_number(abs(factor.impact) * 100)} pontos percentuais em relação à referência SHAP, "
                                 "considerando também as outras informações do registro.")
        source = "conjunto de treino" if context.get("source") == "training" else "amostra de referência do modelo antigo"
        text = f"Referência: {source}, com {context.get('rows', 0)} registros. "
        group = None
        if factor.value is None:
            text += "O valor está ausente; o pipeline usou uma imputação. Confira o dado antes de interpretar o efeito."
        elif item.get("kind") == "numeric":
            value = parse_numeric(pd.Series([factor.value])).iloc[0]
            if pd.notna(value):
                position = "abaixo" if value < item["median"] else "acima" if value > item["median"] else "no mesmo valor"
                text += f"Valor informado: {_number(value)}; mediana histórica: {_number(item['median'])}. O valor está {position} da mediana. "
                if value < item["min"] or value > item["max"]:
                    text += "Está fora da faixa observada no treino; esta comparação é incerta. "
                group = next((b for b in item["bins"] if (b["lower"] is None or value > b["lower"]) and (b["upper"] is None or value <= b["upper"])), None)
                if group:
                    limits = []
                    if group["lower"] is not None:
                        limits.append(f"acima de {_number(group['lower'])}")
                    if group["upper"] is not None:
                        limits.append(f"até {_number(group['upper'])}")
                    text += "Faixa comparada: " + " e ".join(limits) + ". "
                factor.scenario_allowed = scenario_allowed(factor.feature) and item["min"] < item["max"]
                factor.reference_min, factor.reference_max = item["min"], item["max"]
        else:
            group = item.get("categories", {}).get(str(factor.value))
            if group is None:
                text += "Categoria sem referência histórica suficiente nesta explicação. "
        if group and group["rate"] is not None and group["others"]["rate"] is not None:
            text += (f"Nos registros históricos desta faixa/categoria ({group['count']} exemplos), o resultado de risco ocorreu em "
                     f"{_number(group['rate'] * 100)}%, contra {_number(group['others']['rate'] * 100)}% nos demais "
                     f"({group['others']['count']} exemplos). Isso descreve associação, sem isolar o efeito desta variável.")
        elif group:
            text += "Não exibimos uma taxa de resultado: faltam rótulos históricos nesta referência ou há menos de 10 registros em um dos grupos."
        factor.historical_context = text
    explanation.context_version = CONTEXT_VERSION
    return explanation
