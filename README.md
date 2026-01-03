# OncoVision API
OncoVision API is a deep learning–based medical image analysis system that detects cancer from medical images (histopathology / X-ray / MRI depending on dataset) using a Convolutional Neural Network (CNN).
<br />
OncoVision API is a backend-only academic project that performs cancer
detection from medical images using a Convolutional Neural Network
(CNN).

The system accepts file uploads of multiple types, safely converts them
into CNN-compatible images, and performs inference using a pretrained
model.
<br />
```
  --------------------------------------------------
  Key Characteristics
  --------------------------------------------------
  - Backend only (FastAPI) - CNN-based inference (no
  transformers, no LLMs) - Fully offline (no cloud
  APIs) - File-based input only (multipart upload) -
  Supports images and documents - Beginner-friendly
  but architecturally clean - Academic +
  production-aligned design

  --------------------------------------------------
```
# Architecture Overview
Client -> FastAPI Route -> Inference Service -> File Converter -> Image
Preprocessing -> CNN Prediction -> Structured JSON Response

Core Design Principle: The CNN only works on images. All non-image files
are converted into images before inference.

```
  --------------------------------------------------
  Supported Input Types
  --------------------------------------------------
  Images: JPG, JPEG, PNG Documents: PDF, DOCX
  Others: Rejected

  --------------------------------------------------
```
# API Response Format
```
{ “status”: “success”, “file_type”: “pdf”, “images_processed”: 3,
“prediction”: “Cancer Detected”, “confidence”: 0.87 }
```
<img src="https://github.com/notysozu/web-assets/blob/main/OncoVision/flowchart.png?raw=true" alt="Card Image - PC Version" style="width: 100%; max-height: 200px; object-fit: cover; border-radius: 5px;">
<br />

```
backend/
│
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # Settings (paths, model name)
│   │
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Request/Response models
│   │
│   ├── model/
│   │   ├── cnn_model.py        # CNN architecture
│   │   ├── load_model.py       # Load trained model
│   │   └── predict.py          # Prediction logic
│   │
│   ├── utils/
│   │   ├── file_converter.py
│   │   ├── image_utils.py      # Resize, normalize, tensor conversion
│   │   └── logger.py
│   │
│   └── services/
│       └── inference.py        # Business logic wrapper
│
├── model_weights/
│   └── cancer_cnn.h5           # Saved model
│
├── requirements.txt
├── README.md
└── .env
```

# Academic Disclaimer

This project is for educational purposes only. It is not a medical
device and must not be used for real-world diagnosis.