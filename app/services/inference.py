# app/services/inference.py
from typing import Dict

from app.utils.file_converter import convert_file_to_images
from app.utils.image_utils import preprocess_images
from app.model.load_model import load_cnn_model
from app.model.predict import predict_cancer


def run_inference(file_path: str, filename: str) -> Dict:
    """
    Run full inference pipeline on an uploaded file.
    """

    # Load model ONCE per process (singleton inside loader)
    model = load_cnn_model()

    # Step 1: Convert file to images
    images = convert_file_to_images(
        file_path=file_path,
        filename=filename
    )

    # Step 2: Preprocess images for CNN
    image_batch = preprocess_images(images)

    # Step 3: Run CNN prediction
    prediction_result = predict_cancer(model, image_batch)

    # Step 4: Build response payload (NO rounding here yet)
    response = {
        "file_type": filename.split(".")[-1].lower(),
        "images_processed": prediction_result["image_count"],
        "prediction": prediction_result["prediction"],
        "confidence": prediction_result["confidence"],
    }

    return response
