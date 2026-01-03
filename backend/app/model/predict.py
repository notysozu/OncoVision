import numpy as np

from app.config import settings
from app.model.load_model import load_cnn_model


def predict_cancer(image_batch: np.ndarray) -> dict:
    """
    Run CNN inference on a batch of preprocessed images.

    Returns:
    - prediction label (Cancer Detected / No Cancer)
    - confidence score
    """
    model = load_cnn_model()

    # Run model prediction
    predictions = model.predict(image_batch)

    # Flatten predictions to 1D array
    predictions = predictions.flatten()

    # Aggregate results (medical-style conservative approach)
    max_confidence = float(predictions.max())

    if max_confidence >= settings.CONFIDENCE_THRESHOLD:
        label = "Cancer Detected"
    else:
        label = "No Cancer Detected"

    return {
        "prediction": label,
        "confidence": round(max_confidence, 4),
        "image_count": len(predictions)
    }
