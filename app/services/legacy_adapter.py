"""Contratos em português mantidos para clientes anteriores à plataforma."""
from app.schemas.dataset import DatasetProfile
from app.schemas.platform import ModelCard


def legacy_analysis(profile: DatasetProfile) -> dict:
    profiles = {p.name: p for p in profile.column_profiles}

    def candidate(name, confidence=0.9, reason="Atributo sensível sugerido pelo nome."):
        p = profiles[name]
        return {"coluna": name, "score": confidence * 10, "motivos": [reason], "valores_exemplo": [],
                "quantidade_unicos": p.unique_values, "taxa_ausentes": p.missing_percentage / 100}

    targets = [candidate(c.column, c.confidence, c.reason) for c in profile.target_candidates]
    sensitive = [candidate(c) for c in profile.fairness_features]
    numeric = sum(p.dtype == "numeric" for p in profile.column_profiles)
    return {"resumo": {"registros_encontrados": profile.rows, "colunas_encontradas": profile.columns,
             "linhas_vazias_removidas": 0, "colunas_vazias_removidas": 0, "colunas_renomeadas": 0,
             "linhas_duplicadas": profile.duplicates, "celulas_ausentes": round(profile.missing_cells_percentage / 100 * profile.rows * profile.columns),
             "colunas_com_ausentes": sum(p.missing_percentage > 0 for p in profile.column_profiles),
             "colunas_numericas": numeric, "colunas_nao_numericas": profile.columns - numeric},
            "recomendacoes": {"target_recomendado": targets[0] if targets else None,
                              "sensitive_recomendado": sensitive[0] if sensitive else None,
                              "top_targets": targets[:3], "top_sensitive": sensitive[:3],
                              "mensagens": [i.message for i in profile.health.issues]},
            "colunas": [{"coluna": p.name, "tipo_inferido": p.dtype, "valores_ausentes": round(p.missing_percentage / 100 * profile.rows),
                         "taxa_ausentes": p.missing_percentage / 100, "valores_unicos": p.unique_values, "valores_exemplo": []} for p in profile.column_profiles]}


def legacy_fairness(report) -> dict:
    return {**(report.comparisons[0].metrics if report.comparisons else {
        "statistical_parity_difference": None, "disparate_impact": None,
        "equal_opportunity_difference": None, "average_odds_difference": None}), "fairness_score": report.score}


def legacy_training(card: ModelCard, profile: DatasetProfile, distribution: dict) -> dict:
    from app.services.models import obter_nome_modelo
    models = {c.algorithm: {"nome": obter_nome_modelo(c.algorithm), "metricas": c.validation.model_dump(),
                            "fairness": legacy_fairness(c.fairness), "feature_importance": {}} for c in card.comparison}
    def winner(candidate, value):
        return {"tipo": candidate.algorithm, "nome": obter_nome_modelo(candidate.algorithm), "valor": value}
    best = max(card.comparison, key=lambda c: c.validation.accuracy)
    fair = [c for c in card.comparison if c.fairness.score is not None]
    best_fair = max(fair, key=lambda c: c.fairness.score) if fair else None
    return {"model_id": card.id, "modelo": card.algorithm, "modelo_nome": obter_nome_modelo(card.algorithm),
        "modelo_principal": {"tipo": card.algorithm, "nome": obter_nome_modelo(card.algorithm), "criterio": card.config.strategy},
        "metricas": card.metrics.model_dump(), "fairness": legacy_fairness(card.fairness), "feature_importance": card.feature_importance,
        "modelos": models, "comparativo_modelos": {"melhor_acuracia": winner(best, best.validation.accuracy),
            "melhor_fairness": winner(best_fair, best_fair.fairness.score) if best_fair else None,
            "melhor_equilibrio": winner(card.comparison[0], card.comparison[0].score),
            "modelos_com_falha": {k: {"nome": obter_nome_modelo(k), "erro": v} for k, v in card.failed_algorithms.items()},
            "insights": ["Comparação feita na validação; métricas principais calculadas no teste separado."]},
        "dataset": {"total_linhas": profile.rows, "total_colunas": profile.columns, "linhas_apos_limpeza": sum(card.split.values()),
            "linhas_finais": sum(card.split.values()), "treino": card.split["train"], "teste": card.split["test"],
            "features_originais": len(card.features), "features_modelo": len(card.features)},
        "analise_dataset": legacy_analysis(profile),
        "preprocessamento": {"target_binarizado": {v: int(v == card.positive_class) for v in card.classes},
            "target_classe_positiva": card.positive_class, "target_estrategia": "classe_de_risco_vs_demais",
            "sensitive_grupo_privilegiado": "Referências descritivas por grupo", "linhas_descartadas_target_nulo": 0,
            "linhas_descartadas_target_invalido": 0, "linhas_descartadas_sensitive_nulo": 0,
            "valores_ausentes_preenchidos": 0, "distribuicao_target": distribution}}
