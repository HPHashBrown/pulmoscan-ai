"""
predict.py

Image preprocessing and inference logic for the lung X-ray classifier.
Kept separate from app.py so the Flask routes stay thin and this logic
can be tested or reused independently.
"""

import io
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import CLASS_NAMES

# Must exactly mirror the validation/eval transform used in train_model.py
# so inference sees images preprocessed the same way the model was
# evaluated on during training.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


class InvalidImageError(Exception):
    """Raised when an uploaded file cannot be read as a valid image."""
    pass


def allowed_file(filename: str) -> bool:
    """Check whether a filename has a supported image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def load_image(file_bytes: bytes) -> Image.Image:
    """
    Open raw uploaded bytes as a PIL Image, raising a friendly
    InvalidImageError for anything corrupted or unreadable.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # force decode now so truncated/corrupt files fail here
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The uploaded file could not be read as a valid image. "
            "It may be corrupted or in an unsupported format."
        ) from exc


def predict_image(model, device, image: Image.Image) -> dict:
    """
    Run the full preprocessing + inference pipeline on a single PIL image.

    Returns a dict with the predicted class label and confidence
    percentage, e.g. {"prediction": "Suspicious", "confidence": 91.3}
    """
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item()) * 100

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 1),
    }
