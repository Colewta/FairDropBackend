"""Semântica local: somente nomes de colunas, sem chamadas externas."""
import re
import unicodedata
from abc import ABC, abstractmethod

from app.schemas.dataset import ColumnRole as Role


def normalize_name(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


SYNONYMS: dict[Role, set[str]] = {
    Role.IDENTIFIER: {"id", "student_id", "matricula", "ra", "cpf", "email", "uuid", "nome", "name", "nome_completo"},
    Role.TARGET: {"target", "label", "y", "class", "classe", "default_payment_next_month", "status", "situacao", "situacao_final", "status_final", "dropout", "evasao", "evadido", "outcome", "resultado", "enrolled", "graduate"},
    Role.SENSITIVE: {"gender", "sexo", "sex", "genero", "race", "ethnicity", "raca", "etnia", "cor", "age", "idade", "birth_date", "data_nascimento", "income", "salary", "renda", "renda_familiar", "disability", "deficiencia", "pcd", "nationality", "nacionalidade", "religiao", "cep", "bairro", "zipcode"},
    Role.DEMOGRAPHIC: {"gender", "sexo", "sex", "genero", "race", "ethnicity", "raca", "etnia", "cor", "age", "idade", "birth_date", "data_nascimento", "nacionalidade", "nationality"},
    Role.ACADEMIC_PERFORMANCE: {"grade", "nota", "media", "score", "gpa", "reprovacoes"},
    Role.ATTENDANCE: {"attendance", "frequency", "frequencia", "faltas", "presenca"},
    Role.ENGAGEMENT: {"engagement", "engajamento", "acessos", "participacao"},
    Role.FINANCIAL: {"income", "salary", "renda", "renda_familiar", "mensalidade", "bolsa"},
    Role.TEMPORAL: {"date", "data", "ano", "year", "semestre", "birth_date", "data_nascimento"},
}


class SemanticAnalyzer(ABC):
    @abstractmethod
    def analyze_columns(self, schema: list[str]) -> dict[str, list[Role]]:
        """Retorna tags a partir do schema, sem dados pessoais."""


class LocalSemanticAnalyzer(SemanticAnalyzer):
    def __init__(self, synonyms: dict[Role, set[str]] | None = None):
        self.synonyms = SYNONYMS if synonyms is None else synonyms

    def analyze_columns(self, schema: list[str]) -> dict[str, list[Role]]:
        result = {}
        for column in schema:
            name = f"_{normalize_name(column)}_"
            result[column] = [role for role, words in self.synonyms.items()
                              if any(f"_{normalize_name(word)}_" in name for word in words)]
            if normalize_name(column) in {"marital_status", "employment_status", "housing_status"}:
                result[column] = [role for role in result[column] if role != Role.TARGET]
                result[column].append(Role.DEMOGRAPHIC)
        return result
