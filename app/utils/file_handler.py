from pathlib import Path
from uuid import uuid4

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"


def salvar_csv(file):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "dataset.csv").name
    suffix = Path(original_name).suffix or ".csv"
    safe_stem = Path(original_name).stem.replace(" ", "_")[:80] or "dataset"
    file_path = UPLOAD_DIR / f"{safe_stem}_{uuid4().hex[:8]}{suffix}"

    with file_path.open("wb") as output:
        output.write(file.file.read())

    return str(file_path)
