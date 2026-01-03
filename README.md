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
```bash
uvicorn app.main:app --reload
```
<br />
```
  --------------------------------------------------
  Key Characteristics
  --------------------------------------------------
  - Backend only (FastAPI) 
  - CNN-based inference (no transformers, no LLMs) 
  - Fully offline (no cloud APIs) 
  - File-based input only (multipart upload) 
  - Supports images and documents 
  - Beginner-friendly but architecturally clean 
  - Academic + production-aligned design
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
{
 “status”: “success”,
 “file_type”: “pdf”,
 “images_processed”: 3,
 “prediction”: “Cancer Detected”,
 “confidence”: 0.87 
}
```
<img src="https://github.com/notysozu/web-assets/blob/main/OncoVision/flowchart.png?raw=true" alt="Card Image - PC Version" style="width: 100%; max-height: 200px; object-fit: cover; border-radius: 5px;">
<br />

```
OncoVision-API/
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api/
│   │   ├── model/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── config.py
│   │   └── main.py
│   ├── model_weights/
│   └── temp_uploads/
│
├── venv/
├── README.md
└── requirements.txt
```

# Academic Disclaimer

This project is for educational purposes only. It is not a medical
device and must not be used for real-world diagnosis.