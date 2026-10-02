import json
from hashlib import sha256
from pathlib import Path

import joblib

from app.schemas.dataset import DatasetError
from app.schemas.platform import ModelCard
from app.services.repository import Repository, checked_id, new_id


class ArtifactStore:
    """Armazenamento substituível. Aceita apenas IDs gerados pelo servidor."""
    def __init__(self, repository: Repository):
        self.root = repository.root / sha256(repository.institution_id.encode()).hexdigest()[:24]

    def path(self, category: str, identifier: str, name: str) -> Path:
        return self.root / category / checked_id(identifier) / name

    def save(self, category: str, identifier: str, name: str, value) -> None:
        path = self.path(category, identifier, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{new_id()}.tmp")
        try:
            if name.endswith(".json"):
                temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            else:
                joblib.dump(value, temporary, compress=3)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, category: str, identifier: str, name: str):
        path = self.path(category, identifier, name)
        if not path.is_file():
            raise DatasetError("ARTIFACT_NOT_FOUND", "Arquivo persistido indisponível.", 404)
        return joblib.load(path)


class ModelRegistry:
    def __init__(self, repository: Repository):
        self.repository = repository
        self.store = ArtifactStore(repository)

    def save(self, card: ModelCard, model, background, context: dict | None = None) -> None:
        self.store.save("models", card.id, "model.joblib", {"pipeline": model, "background": background, "context": context})
        self.store.save("models", card.id, "metadata.json", card.model_dump(mode="json"))
        self.repository.put("models", card.model_dump(mode="json"))

    def load(self, identifier: str):
        card = ModelCard.model_validate(self.repository.get("models", identifier))
        return card, self.store.load("models", card.id, "model.joblib")

    def delete(self, identifier: str) -> None:
        card = self.repository.get("models", identifier)
        for filename in ("model.joblib", "metadata.json"):
            self.store.path("models", card["id"], filename).unlink(missing_ok=True)
        self.repository.remove("models", identifier)
