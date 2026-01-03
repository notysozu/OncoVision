import os
import tensorflow as tf

from app.config import settings


_model = None  # internal singleton


def load_cnn_model():
    """
    Load the CNN model only once (singleton-style).
    """
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(settings.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {settings.MODEL_PATH}"
        )

    try:
        _model = tf.keras.models.load_model(settings.MODEL_PATH)
        return _model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CNN model: {str(e)}"
        )
