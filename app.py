"""
app.py

Flask application for the AI Lung X-ray Classifier demo.

Routes:
    GET  /            Upload page
    POST /analyze      Server-rendered results page (browser form flow)
    POST /predict       JSON API endpoint
    GET  /about         About / model info page

Inference logic lives in predict.py, model loading lives in model.py.
This file only wires HTTP requests to that logic and handles errors.
"""

import base64
import logging

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)
from werkzeug.exceptions import RequestEntityTooLarge

from model import load_model
from predict import (
    allowed_file,
    load_image,
    predict_image,
    InvalidImageError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# -----------------------
# Load the model once, at startup
# -----------------------
try:
    model, device = load_model()
    logger.info("Lung X-ray model loaded successfully on %s.", device)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001 - want to surface any load failure
    model, device = None, None
    MODEL_LOAD_ERROR = str(exc)
    logger.error("Failed to load model: %s", MODEL_LOAD_ERROR)


EXPLANATIONS = {
    "Normal": (
        "The model did not detect image features resembling a suspicious "
        "mass or nodule in this X-ray. This does NOT guarantee the "
        "absence of disease \u2014 the AI only recognizes patterns similar "
        "to what it saw during training, and it can miss findings a "
        "radiologist would catch."
    ),
    "Suspicious": (
        "The model detected features similar to masses or nodules seen in "
        "its training data. This is not a diagnosis. Please consult a "
        "qualified healthcare professional for a proper evaluation of "
        "this image."
    ),
}


def _read_and_validate_upload():
    """
    Shared validation for both the browser and API upload flows.

    Returns (image, file_bytes, filename) on success.
    Raises ValueError with a user-facing message on failure.
    """
    if "xray" not in request.files:
        raise ValueError("No file was uploaded. Please choose an X-ray image.")

    file = request.files["xray"]

    if file.filename == "":
        raise ValueError("No file was selected. Please choose an X-ray image.")

    if not allowed_file(file.filename):
        raise ValueError(
            "Unsupported file type. Please upload a PNG, JPG, or JPEG image."
        )

    file_bytes = file.read()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        image = load_image(file_bytes)
    except InvalidImageError as exc:
        raise ValueError(str(exc)) from exc

    return image, file_bytes, file.filename


@app.route("/")
def index():
    return render_template("index.html", model_error=MODEL_LOAD_ERROR)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Server-rendered flow used by the upload page's form."""
    if model is None:
        return render_template(
            "index.html",
            model_error=MODEL_LOAD_ERROR,
            upload_error="The model is not available right now. Please try again later.",
        )

    try:
        image, file_bytes, filename = _read_and_validate_upload()
    except ValueError as exc:
        return render_template("index.html", model_error=MODEL_LOAD_ERROR, upload_error=str(exc))

    try:
        result = predict_image(model, device, image)
    except Exception:  # noqa: BLE001
        logger.exception("Inference failed for uploaded file %s", filename)
        return render_template(
            "index.html",
            model_error=MODEL_LOAD_ERROR,
            upload_error="Something went wrong while analyzing this image. Please try again.",
        )

    # Embed the image directly so no file needs to persist on disk.
    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
    mime = "image/png" if filename.lower().endswith("png") else "image/jpeg"
    image_data_uri = f"data:{mime};base64,{encoded_image}"

    return render_template(
        "result.html",
        prediction=result["prediction"],
        confidence=result["confidence"],
        image_data_uri=image_data_uri,
        explanation=EXPLANATIONS[result["prediction"]],
    )


@app.route("/predict", methods=["POST"])
def predict():
    """JSON API endpoint for programmatic access."""
    if model is None:
        return jsonify({"error": "Model is not available on the server."}), 503

    try:
        image, _file_bytes, _filename = _read_and_validate_upload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = predict_image(model, device, image)
    except Exception:  # noqa: BLE001
        logger.exception("Inference failed during /predict")
        return jsonify({"error": "Inference failed. Please try again."}), 500

    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_exc):
    return render_template(
        "index.html",
        model_error=MODEL_LOAD_ERROR,
        upload_error="That file is too large. Please upload an image under 10 MB.",
    ), 413


@app.errorhandler(500)
def handle_server_error(_exc):
    return render_template(
        "index.html",
        model_error=MODEL_LOAD_ERROR,
        upload_error="An unexpected server error occurred. Please try again.",
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
