from __future__ import annotations

import json
from pathlib import Path

import torch

from nsclc_unet.config import PipelineConfig
from nsclc_unet.model import build_feature_extractor, get_best_device


def _serialize_model_config(config: PipelineConfig) -> dict[str, object]:
    return {
        "backbone_name": config.model.backbone_name,
        "in_channels": config.model.in_channels,
        "embedding_pool": config.model.embedding_pool,
        "conv1_t_size": config.model.conv1_t_size,
        "conv1_t_stride": config.model.conv1_t_stride,
        "no_max_pool": config.model.no_max_pool,
        "shortcut_type": config.model.shortcut_type,
        "widen_factor": config.model.widen_factor,
        "bias_downsample": config.model.bias_downsample,
        "weights_path": str(config.model.weights_path) if config.model.weights_path else None,
    }


def prepare_backbone_checkpoint(config: PipelineConfig) -> Path:
    device = get_best_device()
    model = build_feature_extractor(config.model, checkpoint_path=None, device=device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": _serialize_model_config(config),
        },
        config.checkpoint_path,
    )

    summary_path = config.output_dir / "backbone_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint_path": str(config.checkpoint_path),
                "device": str(device),
                "model": _serialize_model_config(config),
                "note": (
                    "This checkpoint snapshots the MONAI 3D ResNet feature extractor. "
                    "If model.weights_path is null, the backbone is randomly initialized."
                ),
            },
            handle,
            indent=2,
        )

    print(f"Saved backbone checkpoint to {config.checkpoint_path}")
    print(f"Saved backbone summary to {summary_path}")
    return config.checkpoint_path


def train_unet(config: PipelineConfig) -> Path:
    return prepare_backbone_checkpoint(config)
