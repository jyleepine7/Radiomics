from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nsclc_unet.config import FeatureConfig, ModelConfig, PipelineConfig, PreprocessingConfig, TabularConfig, TrainingConfig
from nsclc_unet.features import extract_patient_embeddings
from nsclc_unet.manifest import ManifestRecord, read_manifest
from nsclc_unet.train import prepare_backbone_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Colab-friendly smoke test: sample patients and extract MONAI 3D ResNet features."
    )
    parser.add_argument("--manifest", required=True, help="Path to the full manifest CSV.")
    parser.add_argument("--output-dir", default="artifacts/colab_smoke_test", help="Directory for outputs.")
    parser.add_argument("--max-patients", type=int, default=5, help="How many patients to sample from the manifest.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient sampling.")
    parser.add_argument("--feature-batch-size", type=int, default=1, help="Batch size for embedding extraction.")
    parser.add_argument("--backbone-name", default="resnet18", help="MONAI 3D ResNet backbone name.")
    parser.add_argument("--weights-path", default=None, help="Optional pretrained weights checkpoint path.")
    parser.add_argument(
        "--embedding-pool",
        choices=("avg", "avg_max"),
        default="avg",
        help="How to pool the penultimate 3D feature map before writing deep features.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("first", "random"),
        default="random",
        help="Use the first N patients or sample N patients randomly.",
    )
    parser.add_argument("--target-depth", type=int, default=64, help="Resize cropped volumes to this depth.")
    parser.add_argument("--target-size", type=int, default=96, help="Resize height and width to target_size.")
    parser.add_argument("--hu-min", type=int, default=-1000, help="Lower HU clipping bound.")
    parser.add_argument("--hu-max", type=int, default=400, help="Upper HU clipping bound.")
    parser.add_argument("--bbox-margin", type=int, default=8, help="Voxel margin around the tumor bounding box.")
    parser.add_argument("--spacing-z", type=float, default=2.5, help="Target spacing along the depth axis.")
    parser.add_argument("--spacing-y", type=float, default=1.0, help="Target spacing along the height axis.")
    parser.add_argument("--spacing-x", type=float, default=1.0, help="Target spacing along the width axis.")
    return parser


def select_records(records: list[ManifestRecord], max_patients: int, selection_mode: str, seed: int) -> list[ManifestRecord]:
    if max_patients < 1:
        raise ValueError("max_patients must be at least 1.")
    if not records:
        raise ValueError("Manifest must contain at least 1 patient.")
    if len(records) <= max_patients:
        return records

    if selection_mode == "first":
        return records[:max_patients]

    rng = random.Random(seed)
    sampled_ids = set(rng.sample([record.patient_id for record in records], k=max_patients))
    return [record for record in records if record.patient_id in sampled_ids]


def write_subset_manifest(records: list[ManifestRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "image_path", "mask_path"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "patient_id": record.patient_id,
                    "image_path": str(record.image_path),
                    "mask_path": str(record.mask_path),
                }
            )
    return output_path


def write_run_summary(config: PipelineConfig, records: list[ManifestRecord], output_path: Path) -> None:
    payload = {
        "manifest_path": str(config.manifest_path),
        "output_dir": str(config.output_dir),
        "checkpoint_path": str(config.checkpoint_path),
        "feature_output_path": str(config.feature_output_path),
        "patients": [record.patient_id for record in records],
        "model": {
            "backbone_name": config.model.backbone_name,
            "weights_path": str(config.model.weights_path) if config.model.weights_path else None,
            "embedding_pool": config.model.embedding_pool,
        },
        "preprocessing": {
            "target_shape": list(config.preprocessing.target_shape),
            "target_spacing": list(config.preprocessing.target_spacing) if config.preprocessing.target_spacing else None,
            "hu_min": config.preprocessing.hu_min,
            "hu_max": config.preprocessing.hu_max,
            "bbox_margin": config.preprocessing.bbox_margin,
        },
        "features": {
            "batch_size": config.features.batch_size,
            "output_filename": config.features.output_filename,
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_generated_config(config: PipelineConfig) -> None:
    payload = {
        "manifest_path": str(config.manifest_path),
        "output_dir": str(config.output_dir),
        "checkpoint_filename": config.checkpoint_filename,
        "preprocessing": {
            "target_depth": config.preprocessing.target_depth,
            "target_height": config.preprocessing.target_height,
            "target_width": config.preprocessing.target_width,
            "hu_min": config.preprocessing.hu_min,
            "hu_max": config.preprocessing.hu_max,
            "bbox_margin": config.preprocessing.bbox_margin,
            "target_spacing_z": config.preprocessing.target_spacing_z,
            "target_spacing_y": config.preprocessing.target_spacing_y,
            "target_spacing_x": config.preprocessing.target_spacing_x,
        },
        "training": config.training.__dict__,
        "model": {
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
        },
        "features": config.features.__dict__,
        "tabular": {
            "prepared_output_filename": config.tabular.prepared_output_filename,
            "evaluation_output_filename": config.tabular.evaluation_output_filename,
            "predictions_output_filename": config.tabular.predictions_output_filename,
            "radiomics_aggregations": list(config.tabular.radiomics_aggregations),
            "endpoint_windows_months": list(config.tabular.endpoint_windows_months),
        },
    }
    with config.config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_config(args: argparse.Namespace, manifest_path: Path, output_dir: Path) -> PipelineConfig:
    output_dir.mkdir(parents=True, exist_ok=True)
    return PipelineConfig(
        config_path=(output_dir / "colab_smoke_test.generated.json"),
        manifest_path=manifest_path,
        output_dir=output_dir,
        checkpoint_filename="resnet18_backbone.pt",
        preprocessing=PreprocessingConfig(
            target_depth=args.target_depth,
            target_height=args.target_size,
            target_width=args.target_size,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            bbox_margin=args.bbox_margin,
            target_spacing_z=args.spacing_z,
            target_spacing_y=args.spacing_y,
            target_spacing_x=args.spacing_x,
        ),
        training=TrainingConfig(),
        model=ModelConfig(
            backbone_name=args.backbone_name,
            in_channels=1,
            embedding_pool=args.embedding_pool,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            shortcut_type="A",
            widen_factor=1.0,
            bias_downsample=True,
            weights_path=Path(args.weights_path).resolve() if args.weights_path else None,
        ),
        features=FeatureConfig(
            batch_size=args.feature_batch_size,
            output_filename="deep_features.csv",
        ),
        tabular=TabularConfig(),
    )


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    records = read_manifest(manifest_path)
    selected_records = select_records(records, args.max_patients, args.selection_mode, args.seed)

    output_dir = Path(args.output_dir).resolve()
    subset_manifest_path = write_subset_manifest(selected_records, output_dir / "manifest_subset.csv")
    config = build_config(args, subset_manifest_path, output_dir)
    write_generated_config(config)
    write_run_summary(config, selected_records, output_dir / "run_summary.json")

    print(f"Selected {len(selected_records)} patients: {[record.patient_id for record in selected_records]}")
    print(f"Subset manifest: {subset_manifest_path}")

    checkpoint_path = prepare_backbone_checkpoint(config)
    feature_path = extract_patient_embeddings(config, checkpoint_path=checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Deep features saved to: {feature_path}")


if __name__ == "__main__":
    main()
