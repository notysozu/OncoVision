import os
from dotenv import load_dotenv
load_dotenv()
class Settings:
    """
    Central configuration for OncoVision API
    """
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "model_weights/cancer_cnn.h5"
    )

    IMAGE_SIZE = (
        int(os.getenv("IMAGE_WIDTH", 128)),
        int(os.getenv("IMAGE_HEIGHT", 128))
    )

    # -----------------------------
    # File Handling Configuration
    # -----------------------------
    ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
    ALLOWED_DOCUMENT_TYPES = ["pdf", "docx"]
    ALLOWED_FILE_TYPES = ALLOWED_IMAGE_TYPES + ALLOWED_DOCUMENT_TYPES
    CONFIDENCE_THRESHOLD = float(
        os.getenv("CONFIDENCE_THRESHOLD", 0.5)
    )
    MAX_UPLOAD_SIZE_MB = int(
        os.getenv("MAX_UPLOAD_SIZE_MB", 20)
    )
settings = Settings()
