"""FastAPI app that serves predictions from a trained checkpoint.

The checkpoint is loaded once at startup from CHECKPOINT_PATH (a mounted volume
locally, the checkpoints PVC on Kubernetes). If it is missing the app still
starts but `/health` returns 503 - this keeps the container up in a "cold"
state (e.g. before training has finished) instead of crash-looping, which is
what the CI smoke test and the readiness probe rely on.

`POST /predict` takes an uploaded image, resizes it to 32x32, applies the same
normalization used in evaluation, and returns per-class probabilities.
"""

import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from dataset import get_transforms
from model import get_model

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"
)

app = FastAPI(title="cifar10-classifier")
_state = {"model": None}
_transform = get_transforms(train=False)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    path = Path(CHECKPOINT_PATH)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=_device)
    model = get_model(
        architecture=ckpt.get("architecture", "resnet18"),
        num_classes=ckpt.get("num_classes", 10),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(_device).eval()
    return model


@app.on_event("startup")
def _startup():
    _state["model"] = load_model()


@app.get("/health")
def health():
    if _state["model"] is None:
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    model = _state["model"]
    if model is None:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32))
    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    scores = {cls: round(float(p), 4) for cls, p in zip(CIFAR10_CLASSES, probs)}
    top = max(scores, key=scores.get)
    return {"prediction": top, "probabilities": scores}
