# PulmoScan AI — Lung X-ray Classifier (Demo)

An educational Flask web app that serves a DenseNet121 chest X-ray classifier
(Normal vs. Suspicious). **Not FDA approved. Not a medical device. Not for
clinical use.**

## Project structure

```
lung-ai-web/
├── app.py              # Flask routes (thin — delegates to model.py / predict.py)
├── model.py             # Loads DenseNet121 + trained weights
├── predict.py           # Preprocessing, validation, and inference logic
├── lung_model.pth       # Trained model weights (already included)
├── requirements.txt
├── templates/
│   ├── base.html         # Shared header, disclaimer banner, footer
│   ├── index.html        # Upload page (drag & drop, preview, spinner)
│   ├── result.html        # Results page
│   └── about.html         # Model card / evaluation metrics
├── static/
│   └── style.css
└── uploads/               # Unused at runtime (images are processed in memory)
```

## Setup

```bash
cd lung-ai-web
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

Then open http://127.0.0.1:5000 in your browser.

## API

**POST /predict** — multipart form upload, field name `xray` (PNG/JPG/JPEG).

```bash
curl -X POST http://127.0.0.1:5000/predict -F "xray=@chest_xray.png"
```

Response:

```json
{"prediction": "Suspicious", "confidence": 91.3}
```

## Notes

- Uploaded images are processed in memory and never written to disk — the
  `uploads/` folder is unused by the app itself, kept only for the requested
  project layout.
- The model is loaded once at server startup (`app.py`), not per-request.
- Evaluation metrics shown on the About page come from the held-out test
  set results (confusion matrix + ROC curve) produced by `train_model.py`.
