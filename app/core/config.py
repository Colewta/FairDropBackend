"""Limites e critérios explícitos da análise (variáveis de ambiente opcionais)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MAX_UPLOAD_BYTES = int(os.getenv("FAIRDROP_MAX_UPLOAD_BYTES", 100 * 1024 * 1024))
MAX_ROWS = int(os.getenv("FAIRDROP_MAX_ROWS", 500_000))
MAX_COLUMNS = int(os.getenv("FAIRDROP_MAX_COLUMNS", 500))
MIN_ASSOCIATION_ROWS = 30
NEAR_PERFECT_CORRELATION = 0.98
IDENTIFIER_RATIO = 0.95
MIN_IDENTIFIER_ROWS = 20
IMBALANCE_RATIO = 0.10

RANDOM_STATE = 42
STORAGE_DIR = Path(os.getenv("FAIRDROP_STORAGE_DIR", str(Path(__file__).resolve().parents[2] / "data" / "platform")))
INSTITUTION_ID = os.getenv("FAIRDROP_INSTITUTION_ID", "local")
API_TOKEN = os.getenv("FAIRDROP_API_TOKEN", "")
RISK_THRESHOLDS = {"low": 0.30, "medium": 0.70}
MODEL_SELECTION_WEIGHTS = {
    "best_predictive": {"recall": 0.4, "roc_auc": 0.3, "pr_auc": 0.3},
    "best_fairness": {"fairness": 0.7, "recall": 0.15, "pr_auc": 0.15},
    "best_balanced": {"recall": 0.3, "roc_auc": 0.2, "pr_auc": 0.15, "fairness": 0.2, "calibration": 0.15},
}
PREDICTION_CHUNK_SIZE = 2000
SHAP_BACKGROUND_ROWS = 12
SHAP_GLOBAL_ROWS = 16
SHAP_MAX_FEATURES = 100
MODEL_VERSION = "2.1"
