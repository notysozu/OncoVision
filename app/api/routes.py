import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.services.inference import run_inference
from app.utils.logger import log
from app.utils.file_converter import (
    UnsupportedFileTypeError,
    FileConversionError
)

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a file upload and run cancer detection.
    """

    # -----------------------------
    # Basic validation
    # -----------------------------
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )

    log.info(f"Received file: {file.filename}")

    # -----------------------------
    # File size validation
    # -----------------------------
    file.file.seek(0, os.SEEK_END)
    file_size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        log.warning(
            f"File too large: {file.filename} ({file_size_mb:.2f} MB)"
        )
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds size limit"
        )

    # -----------------------------
    # Generate safe unique filename
    # -----------------------------
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # -----------------------------
        # Save uploaded file
        # -----------------------------
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log.info(f"File saved to disk: {file_path}")

        # -----------------------------
        # Run inference pipeline
        # -----------------------------
        result = run_inference(
            file_path=file_path,
            filename=file.filename
        )

        # -----------------------------
        # Presentation-layer formatting
        # -----------------------------
        response = {
            "status": "success",
            "file_type": result["file_type"],
            "images_processed": result["images_processed"],
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 6)
        }

        log.info(
            f"Inference completed | "
            f"Prediction={response['prediction']} | "
            f"Confidence={response['confidence']}"
        )

        return response

    except UnsupportedFileTypeError as e:
        log.warning(str(e))
        raise HTTPException(
            status_code=415,
            detail=str(e)
        )

    except FileConversionError as e:
        log.error(str(e))
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        log.exception("Unexpected inference failure")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference"
        )

    finally:
        # -----------------------------
        # Cleanup temporary file
        # -----------------------------
        if os.path.exists(file_path):
            os.remove(file_path)
            log.info(f"Cleaned up file: {file_path}")
