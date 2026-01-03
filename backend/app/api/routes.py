import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.logger import log
from app.config import settings
from app.services.inference import run_inference
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

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save uploaded file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference pipeline
        result = run_inference(
            file_path=file_path,
            filename=file.filename
        )

        return {
            "status": "success",
            **result
        }

    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=415,
            detail=str(e)
        )

    except FileConversionError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
