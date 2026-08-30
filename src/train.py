"""Training entry point.

Reads hyperparameters from a YAML config, logs metrics as JSON lines to
stdout, saves the best checkpoint, and supports early stopping.

Config path resolution order:
  1. TRAINING_CONFIG env var
  2. /app/configs/training_config.yaml  (mounted ConfigMap)
  3. configs/training_config.yaml       (local run)
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def log(entry: dict) -> None:
    print(json.dumps(entry), flush=True)


def load_config() -> dict:
    candidates = [
        os.environ.get("TRAINING_CONFIG"),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
    ]
    for path in candidates:
        if path and Path(path).exists():
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("no training config found")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "start", "device": str(device)})

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            log({"event": "checkpoint_saved", "path": str(save_path)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log({"event": "early_stopping", "epoch": epoch + 1})
                break

    log({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})


if __name__ == "__main__":
    main()
