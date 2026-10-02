"""Persistência de domínio; SQLAlchemy permite substituir SQLite posteriormente."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4
from itertools import islice
from collections.abc import Iterable

from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, create_engine, delete, func, select, update

from app.core import config
from app.schemas.dataset import DatasetError

metadata = MetaData()
resources = {
    name: Table(name, metadata, Column("id", String, primary_key=True), Column("institution_id", String, index=True),
                Column("created_at", String, index=True), Column("payload", JSON, nullable=False))
    for name in ("institutions", "datasets", "training_runs", "models", "prediction_batches", "tutor_reviews")
}
predictions = Table("student_predictions", metadata,
    Column("id", String, primary_key=True), Column("institution_id", String, index=True),
    Column("batch_id", String, index=True), Column("student_id", String, index=True),
    Column("created_at", String, index=True), Column("probability", Float, index=True),
    Column("risk_level", String, index=True), Column("payload", JSON), Column("features", JSON), Column("explanation", JSON))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid4().hex


def checked_id(value: str) -> str:
    try:
        return UUID(value).hex
    except (ValueError, AttributeError) as exc:
        raise DatasetError("NOT_FOUND", "Registro não encontrado.", 404) from exc


class Repository:
    def __init__(self, root: Path, institution_id: str):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.institution_id = institution_id
        self.engine = create_engine(f"sqlite:///{(self.root / 'fairdrop.sqlite3').as_posix()}", connect_args={"timeout": 30})
        metadata.create_all(self.engine)

    def put(self, kind: str, payload: dict) -> None:
        table = resources[kind]
        # JSON roundtrip rejects non-finite numbers before database writes.
        json.dumps(payload, allow_nan=False)
        with self.engine.begin() as connection:
            exists = connection.execute(select(table.c.id).where(table.c.id == payload["id"], table.c.institution_id == self.institution_id)).first()
            if exists:
                connection.execute(update(table).where(table.c.id == payload["id"], table.c.institution_id == self.institution_id).values(payload=payload))
            else:
                connection.execute(table.insert().values(id=payload["id"], institution_id=self.institution_id,
                                                        created_at=payload.get("created_at", now()), payload=payload))

    def get(self, kind: str, identifier: str) -> dict:
        table = resources[kind]
        with self.engine.connect() as connection:
            result = connection.execute(select(table.c.payload).where(table.c.id == checked_id(identifier), table.c.institution_id == self.institution_id)).scalar_one_or_none()
        if result is None:
            raise DatasetError("NOT_FOUND", "Registro não encontrado.", 404)
        return result

    def list(self, kind: str, limit: int = 50, offset: int = 0) -> list[dict]:
        table = resources[kind]
        with self.engine.connect() as connection:
            return list(connection.execute(select(table.c.payload).where(table.c.institution_id == self.institution_id)
                                           .order_by(table.c.created_at.desc()).limit(limit).offset(offset)).scalars())

    def remove(self, kind: str, identifier: str) -> None:
        self.get(kind, identifier)
        table = resources[kind]
        with self.engine.begin() as connection:
            connection.execute(delete(table).where(table.c.id == checked_id(identifier), table.c.institution_id == self.institution_id))

    def save_batch(self, batch: dict, rows: Iterable[dict]) -> None:
        table = resources["prediction_batches"]
        with self.engine.begin() as connection:
            connection.execute(table.insert().values(id=batch["id"], institution_id=self.institution_id, created_at=batch["created_at"], payload=batch))
            iterator = iter(rows)
            while chunk := list(islice(iterator, 1000)):
                connection.execute(predictions.insert(), chunk)

    def prediction_page(self, batch_id: str, page: int, size: int, risk: str | None, query: str = "",
                        group_column: str | None = None, group_value: str | None = None) -> tuple[int, list[dict]]:
        batch = self.get("prediction_batches", batch_id)
        conditions = [predictions.c.institution_id == self.institution_id, predictions.c.batch_id == checked_id(batch_id)]
        if risk:
            conditions.append(predictions.c.risk_level == risk)
        if query:
            conditions.append(predictions.c.student_id.contains(query, autoescape=True))
        if group_column:
            if group_column not in batch.get("group_filters", {}):
                raise DatasetError("INVALID_GROUP_FILTER", "Esta base não possui o filtro de curso ou turma informado.")
            conditions.append(predictions.c.payload["groups"][group_column].as_string() == group_value)
        with self.engine.connect() as connection:
            total = connection.execute(select(func.count()).select_from(predictions).where(*conditions)).scalar_one()
            items = connection.execute(select(predictions.c.payload).where(*conditions).order_by(predictions.c.probability.desc(), predictions.c.id).limit(size).offset((page - 1) * size)).scalars().all()
        return total, items

    def student(self, identifier: str) -> dict:
        with self.engine.connect() as connection:
            row = connection.execute(select(predictions).where(predictions.c.id == checked_id(identifier), predictions.c.institution_id == self.institution_id)).mappings().first()
        if row is None:
            raise DatasetError("NOT_FOUND", "Previsão não encontrada.", 404)
        return dict(row)

    def history(self, student_id: str) -> tuple[int, list[dict]]:
        conditions = [predictions.c.institution_id == self.institution_id, predictions.c.student_id == student_id]
        with self.engine.connect() as connection:
            total = connection.execute(select(func.count()).select_from(predictions).where(*conditions)).scalar_one()
            items = connection.execute(select(predictions.c.payload).where(*conditions).order_by(predictions.c.created_at.desc()).limit(100)).scalars().all()
        return total, items

    def save_explanation(self, identifier: str, explanation: dict, payload: dict) -> None:
        with self.engine.begin() as connection:
            connection.execute(update(predictions).where(predictions.c.id == checked_id(identifier), predictions.c.institution_id == self.institution_id).values(explanation=explanation, payload=payload))

    def delete_batch(self, identifier: str) -> None:
        self.get("prediction_batches", identifier)
        with self.engine.begin() as connection:
            ids = select(predictions.c.id).where(predictions.c.batch_id == checked_id(identifier), predictions.c.institution_id == self.institution_id)
            reviews = resources["tutor_reviews"]
            connection.execute(delete(reviews).where(reviews.c.id.in_(ids), reviews.c.institution_id == self.institution_id))
            connection.execute(delete(predictions).where(predictions.c.batch_id == checked_id(identifier), predictions.c.institution_id == self.institution_id))
        self.remove("prediction_batches", identifier)


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    repository = Repository(config.STORAGE_DIR, config.INSTITUTION_ID)
    return repository
