from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nsclc_unet.config import PipelineConfig
from nsclc_unet.io import load_case_with_metadata, save_feature_rows
from nsclc_unet.manifest import read_manifest
from nsclc_unet.model import build_feature_extractor, get_best_device
from nsclc_unet.preprocess import prepare_tumor_volume


def _rows_from_batch(patient_ids: list[str], embeddings: np.ndarray) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for patient_id, embedding in zip(patient_ids, embeddings):
        row: dict[str, str | float] = {"patient_id": patient_id}
        for index, value in enumerate(embedding.tolist()):
            row[f"deep_feat_{index:03d}"] = float(value)
        rows.append(row)
    return rows


def _flush_volume_batch(
    patient_ids: list[str],
    pending_volumes: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, str | float]]:
    if not patient_ids:
        return []

    volume_batch = np.stack(pending_volumes, axis=0).astype(np.float32)
    volume_tensor = torch.from_numpy(volume_batch[:, None, ...]).float().to(device)
    with torch.no_grad():
        embeddings = model(volume_tensor).detach().cpu().numpy()
    return _rows_from_batch(patient_ids, embeddings)


def extract_patient_embeddings(config: PipelineConfig, checkpoint_path: Path | None = None) -> Path:
    records = read_manifest(config.manifest_path)
    device = get_best_device()
    checkpoint = checkpoint_path or config.checkpoint_path
    checkpoint_override = checkpoint if checkpoint.exists() else None
    model = build_feature_extractor(config.model, checkpoint_path=checkpoint_override, device=device)

    output_rows: list[dict[str, str | float]] = []
    pending_ids: list[str] = []
    pending_volumes: list[np.ndarray] = []

    for record in records:
        image, mask, spacing = load_case_with_metadata(record.image_path, record.mask_path)
        volume, _ = prepare_tumor_volume(image, mask, config.preprocessing, spacing=spacing)
        pending_ids.append(record.patient_id)
        pending_volumes.append(volume)

        if len(pending_volumes) >= max(1, config.features.batch_size):
            output_rows.extend(_flush_volume_batch(pending_ids, pending_volumes, model, device))
            pending_ids = []
            pending_volumes = []

    output_rows.extend(_flush_volume_batch(pending_ids, pending_volumes, model, device))
    save_feature_rows(output_rows, config.feature_output_path)
    print(f"Saved deep features to {config.feature_output_path}")
    return config.feature_output_path
