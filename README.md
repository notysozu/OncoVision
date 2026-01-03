# OncoVision API
OncoVision API is a deep learning–based medical image analysis system that detects cancer from medical images (histopathology / X-ray / MRI depending on dataset) using a Convolutional Neural Network (CNN).
<br />
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

