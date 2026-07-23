"""
model.py

Handles loading of the trained DenseNet121 lung X-ray classifier.
Kept separate from Flask routes and inference logic so the model
lifecycle (loading, device placement) has a single, clear home.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

# On constrained/shared-vCPU hosts (e.g. free-tier cloud instances),
# PyTorch's default of using every detected CPU core for intra-op
# parallelism can badly backfire: the host may only actually grant a
# small fraction of a core, so spinning up many threads causes
# contention/context-switching overhead rather than speedup. A single
# small 224x224 image through DenseNet121 doesn't benefit much from
# multi-threading anyway, so pin it down for consistently fast, low
# overhead inference regardless of host.
torch.set_num_threads(1)

# Class order must match the ImageFolder class_to_idx mapping used
# during training (alphabetical: "normal" -> 0, "suspicious" -> 1).
CLASS_NAMES = ["Normal", "Suspicious"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lung_model.pth")


def get_device():
    """Return the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str = MODEL_PATH, device: torch.device = None):
    """
    Build a DenseNet121 with a 2-class head and load the trained
    weights from disk.

    ImageNet weights are intentionally disabled (weights=None) because
    lung_model.pth already contains fully trained weights for this task.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find trained model weights at '{model_path}'. "
            "Make sure lung_model.pth is present in the project root."
        )

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, device
