"""
predict.py

Image preprocessing and inference logic for the lung X-ray classifier.
Kept separate from app.py so the Flask routes stay thin and this logic
can be tested or reused independently.
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
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

# Same resize/grayscale as above but no normalization — used as the plain
# RGB base image that the Grad-CAM heatmap gets blended on top of.
DISPLAY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
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


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """
    Map a 2D array of values in [0, 1] to an RGB heat-style colormap
    (blue -> cyan -> green -> yellow -> red), without needing matplotlib.
    Returns an array of shape (H, W, 3) with values in [0, 1].
    """
    r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam(model, device, image: Image.Image) -> str:
    """
    Run Grad-CAM against the predicted class and return a base64 PNG
    data URI of the original X-ray with a heatmap overlay showing which
    regions most influenced the model's prediction.

    The forward pass is reconstructed manually (features -> ReLU -> pool
    -> classifier) rather than calling model(tensor) directly, because
    DenseNet121's built-in forward uses an in-place ReLU that's
    incompatible with capturing gradients on the feature map.
    """
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    activation = model.features(tensor)      # (1, C, 7, 7)
    activation.retain_grad()

    relu_out = F.relu(activation)
    pooled = F.adaptive_avg_pool2d(relu_out, (1, 1))
    pooled = torch.flatten(pooled, 1)
    logits = model.classifier(pooled)

    predicted_index = int(torch.argmax(logits, dim=1).item())
    score = logits[0, predicted_index]
    score.backward()

    gradient = activation.grad[0]             # (C, H, W)
    activation_values = activation[0].detach()  # (C, H, W)

    weights = gradient.mean(dim=(1, 2))        # (C,)
    cam = torch.relu((weights[:, None, None] * activation_values).sum(dim=0))

    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsample the (7, 7) CAM to full image resolution.
    cam_tensor = torch.tensor(cam)[None, None, :, :]
    cam_resized = F.interpolate(
        cam_tensor, size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    heatmap_rgb = _colorize(cam_resized)  # (224, 224, 3) in [0, 1]

    base_image = np.array(DISPLAY_TRANSFORM(image)).astype(np.float32) / 255.0
    alpha = 0.45
    blended = base_image * (1 - alpha) + heatmap_rgb * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    overlay_image = Image.fromarray(blended)

    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
