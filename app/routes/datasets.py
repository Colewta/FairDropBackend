import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.dataset import DatasetError, DatasetProfile
from app.services.dataset_profiler import DatasetProfiler
from app.services.dataset_reader import read_upload

router = APIRouter(tags=["datasets"])
logger = logging.getLogger(__name__)


@router.post("/analyze-dataset", response_model=DatasetProfile)
def analyze_dataset(file: UploadFile = File(...), target: str | None = Form(None)) -> DatasetProfile:
    """Analisa sem treinar; target opcional permite recalcular após revisão."""
    try:
        df = read_upload(file)
        profile = DatasetProfiler().profile(df, target=target)
        logger.info("Dataset analisado: rows=%d columns=%d", profile.rows, profile.columns)
        return profile
    except DatasetError as exc:
        raise HTTPException(status_code=exc.status_code,
                            detail={"error": exc.code, "message": str(exc)}) from exc
    except Exception as exc:
        # Não registrar mensagem/traceback de parsers que podem conter células pessoais.
        logger.error("Falha na análise: exception_type=%s", type(exc).__name__)
        raise HTTPException(status_code=500, detail={
            "error": "ANALYSIS_FAILED", "message": "Não foi possível concluir a análise da base."
        }) from exc
