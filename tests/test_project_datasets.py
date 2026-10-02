"""Regressão com as bases locais quando presentes; sem imprimir dados individuais."""
from pathlib import Path

import pytest

from app.services.dataset_reader import read_dataset
from app.services.dataset_profiler import DatasetProfiler

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
FILES = [path for path in DATA_ROOT.rglob("*.csv") if "platform" not in path.parts and "uploads" not in path.parts]


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_real_project_files_are_readable(path):
    frame = read_dataset(path.read_bytes(), path.name)
    profile = DatasetProfiler().profile(frame)
    assert len(frame) > 0 and len(frame.columns) > 1
    if path.name.startswith("bank"):
        assert profile.selected_target == "y"
    if path.name == "default of credit card clients.csv":
        assert frame.shape == (30000, 25)
        assert profile.import_info["format"] == "XLS"
