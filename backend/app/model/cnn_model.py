from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def build_cnn(input_shape):
    """
    Reference CNN architecture used for training.
    This is NOT used during inference directly if loading a .h5 model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    return model
