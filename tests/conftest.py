import pytest

from app.core import config
from app.services.repository import get_repository


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("FAIRDROP_DEMO_ENABLED", "false")
    get_repository.cache_clear()
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path / "platform")
    repository = get_repository()
    yield repository
    repository.engine.dispose()
    get_repository.cache_clear()
