from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nsclc_unet.config import PipelineConfig
from nsclc_unet.dataset import CTSliceDataset
from nsclc_unet.manifest import ManifestRecord, read_manifest
from nsclc_unet.model import UNet2D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(records: list[ManifestRecord], validation_fraction: float, seed: int) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    patient_ids = sorted({record.patient_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    val_count = max(1, int(round(len(patient_ids) * validation_fraction)))
    val_ids = set(patient_ids[:val_count])
    train_ids = set(patient_ids[val_count:])
    if not train_ids:
        raise ValueError("Validation split consumed all patients. Reduce validation_fraction.")

    train_records = [record for record in records if record.patient_id in train_ids]
    val_records = [record for record in records if record.patient_id in val_ids]
    return train_records, val_records


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice_loss(logits, targets)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float().flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return float(dice.mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, masks)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_dice += dice_score(logits.detach(), masks)
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "dice": total_dice / max(total_batches, 1),
    }


def train_unet(config: PipelineConfig) -> Path:
    set_seed(config.training.random_seed)
    records = read_manifest(config.manifest_path)
    train_records, val_records = split_records(records, config.training.validation_fraction, config.training.random_seed)

    train_dataset = CTSliceDataset(
        train_records,
        preprocessing=config.preprocessing,
        augment=True,
        random_seed=config.training.random_seed,
    )
    val_dataset = CTSliceDataset(
        val_records,
        preprocessing=config.preprocessing,
        augment=False,
        random_seed=config.training.random_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config.model.__dict__,
                    "preprocessing_config": config.preprocessing.__dict__,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                config.checkpoint_path,
            )

        if epoch - best_epoch >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    history_path = config.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "train_patients": [record.patient_id for record in train_records],
                "val_patients": [record.patient_id for record in val_records],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            handle,
            indent=2,
        )

    print(f"Saved best checkpoint to {config.checkpoint_path}")
    print(f"Saved training history to {history_path}")
    return config.checkpoint_path

