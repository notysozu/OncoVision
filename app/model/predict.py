# app/model/predict.py
import numpy as np
from app.config import settings


def predict_cancer(model, image_batch: np.ndarray) -> dict:
    """
    Pure prediction logic. No model loading. No rounding.
    """
    predictions = model.predict(image_batch)
    predictions = predictions.flatten()

    max_confidence = float(predictions.max())

    label = (
        "Cancer Detected"
        if max_confidence >= settings.CONFIDENCE_THRESHOLD
        else "No Cancer Detected"
    )

    return {
        "prediction": label,
        "confidence": max_confidence,
        "image_count": len(predictions)
    }
