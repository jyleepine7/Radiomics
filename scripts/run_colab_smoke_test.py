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
from nsclc_unet.train import train_unet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Colab-friendly smoke test: sample patients, train UNet, and extract deep features."
    )
    parser.add_argument("--manifest", required=True, help="Path to the full manifest CSV.")
    parser.add_argument("--output-dir", default="artifacts/colab_smoke_test", help="Directory for outputs.")
    parser.add_argument("--max-patients", type=int, default=5, help="How many patients to sample from the manifest.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient sampling and training.")
    parser.add_argument("--epochs", type=int, default=5, help="Short training schedule for Colab validation.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--feature-batch-size", type=int, default=8, help="Batch size for embedding extraction.")
    parser.add_argument("--base-channels", type=int, default=8, help="Smaller UNet width for quicker smoke tests.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the UNet blocks.")
    parser.add_argument(
        "--embedding-pool",
        choices=("avg", "avg_max"),
        default="avg",
        help="How to pool the bottleneck before writing deep features.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("first", "random"),
        default="random",
        help="Use the first N patients or sample N patients randomly.",
    )
    parser.add_argument("--target-size", type=int, default=256, help="Resize tumor slices to target_size x target_size.")
    parser.add_argument("--hu-min", type=int, default=-1000, help="Lower HU clipping bound.")
    parser.add_argument("--hu-max", type=int, default=400, help="Upper HU clipping bound.")
    parser.add_argument("--bbox-margin", type=int, default=16, help="Voxel margin around the tumor bounding box.")
    parser.add_argument("--min-mask-pixels", type=int, default=8, help="Minimum positive pixels for keeping a slice.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction at the patient level. With 5 patients this usually becomes 1 validation patient.",
    )
    return parser


def select_records(records: list[ManifestRecord], max_patients: int, selection_mode: str, seed: int) -> list[ManifestRecord]:
    if max_patients < 2:
        raise ValueError("max_patients must be at least 2 so training can keep at least one validation patient.")
    if len(records) < 2:
        raise ValueError("Manifest must contain at least 2 patients.")
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
        "training": config.training.__dict__,
        "model": config.model.__dict__,
        "preprocessing": config.preprocessing.__dict__,
        "features": config.features.__dict__,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_generated_config(config: PipelineConfig) -> None:
    payload = {
        "manifest_path": str(config.manifest_path),
        "output_dir": str(config.output_dir),
        "checkpoint_filename": config.checkpoint_filename,
        "preprocessing": config.preprocessing.__dict__,
        "training": config.training.__dict__,
        "model": config.model.__dict__,
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
        checkpoint_filename="unet_best.pt",
        preprocessing=PreprocessingConfig(
            target_height=args.target_size,
            target_width=args.target_size,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            min_mask_pixels=args.min_mask_pixels,
            bbox_margin=args.bbox_margin,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=1e-3,
            weight_decay=1e-4,
            validation_fraction=args.val_fraction,
            random_seed=args.seed,
            num_workers=0,
            early_stopping_patience=max(2, min(args.epochs, 4)),
        ),
        model=ModelConfig(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            dropout=args.dropout,
            embedding_pool=args.embedding_pool,
        ),
        features=FeatureConfig(
            batch_size=args.feature_batch_size,
            aggregate_mean=True,
            aggregate_max=True,
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

    checkpoint_path = train_unet(config)
    feature_path = extract_patient_embeddings(config, checkpoint_path=checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Deep features saved to: {feature_path}")


if __name__ == "__main__":
    main()
