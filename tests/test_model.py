"""Model shape/sanity tests - pure torch, no data files or checkpoint needed,
so they run in CI without any download or trained model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import get_model  # noqa: E402


def test_resnet18_output_shape():
    model = get_model("resnet18", num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_smallcnn_output_shape():
    model = get_model("smallcnn", num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_unknown_architecture_raises():
    try:
        get_model("does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown architecture")
