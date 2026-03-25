from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nsclc_unet.config import PipelineConfig
from nsclc_unet.io import load_case, save_feature_rows
from nsclc_unet.manifest import read_manifest
from nsclc_unet.model import UNet2D
from nsclc_unet.preprocess import extract_tumor_slices


def _build_model_from_checkpoint(config: PipelineConfig, checkpoint_path: Path, device: torch.device) -> UNet2D:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet2D(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _aggregate_embeddings(embeddings: np.ndarray, include_mean: bool, include_max: bool) -> np.ndarray:
    parts = []
    if include_mean:
        parts.append(embeddings.mean(axis=0))
    if include_max:
        parts.append(embeddings.max(axis=0))
    if not parts:
        raise ValueError("At least one aggregation mode must be enabled.")
    return np.concatenate(parts, axis=0)


def extract_patient_embeddings(config: PipelineConfig, checkpoint_path: Path | None = None) -> Path:
    records = read_manifest(config.manifest_path)
    checkpoint = checkpoint_path or config.checkpoint_path
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model_from_checkpoint(config, checkpoint, device)

    output_rows: list[dict[str, str | float]] = []
    for record in records:
        image, mask = load_case(record.image_path, record.mask_path)
        slices = extract_tumor_slices(image, mask, config.preprocessing)
        image_batch = np.stack([image_slice for image_slice, _ in slices], axis=0).astype(np.float32)
        image_tensor = torch.from_numpy(image_batch[:, None, ...]).float()

        dataset = TensorDataset(image_tensor)
        loader = DataLoader(
            dataset,
            batch_size=config.features.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        batch_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for (images,) in loader:
                images = images.to(device)
                embeddings = model.extract_embedding(images).cpu().numpy()
                batch_embeddings.append(embeddings)

        patient_embeddings = np.concatenate(batch_embeddings, axis=0)
        aggregated = _aggregate_embeddings(
            patient_embeddings,
            include_mean=config.features.aggregate_mean,
            include_max=config.features.aggregate_max,
        )

        row: dict[str, str | float] = {"patient_id": record.patient_id}
        for index, value in enumerate(aggregated.tolist()):
            row[f"deep_feat_{index:03d}"] = float(value)
        output_rows.append(row)

    save_feature_rows(output_rows, config.feature_output_path)
    print(f"Saved deep features to {config.feature_output_path}")
    return config.feature_output_path

