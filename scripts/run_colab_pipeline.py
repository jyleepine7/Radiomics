from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nsclc_unet.config import FeatureConfig, ModelConfig, PipelineConfig, PreprocessingConfig, TabularConfig, TrainingConfig
from nsclc_unet.features import extract_patient_embeddings
from nsclc_unet.manifest import ManifestRecord, read_manifest
from nsclc_unet.modeling import fit_endpoint_models
from nsclc_unet.plotting import plot_endpoint_roc_curves
from nsclc_unet.tabular import prepare_tabular_dataset
from nsclc_unet.train import prepare_backbone_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MONAI 3D ResNet pipeline end-to-end in Google Colab."
    )
    parser.add_argument("--manifest", required=True, help="Path to the patient manifest CSV.")
    parser.add_argument("--xlsx-path", required=True, help="Path to Radiomics_Clinical.xlsx.")
    parser.add_argument("--weights-path", required=True, help="Path to the MedicalNet/Med3D resnet18 checkpoint.")
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/Radiomics_colab_data/output",
        help="Directory where extracted features, prepared tables, and metrics will be written.",
    )
    parser.add_argument(
        "--generated-config",
        default=None,
        help="Optional path for the generated Colab config JSON. Defaults to output_dir/colab_pipeline.generated.json.",
    )
    parser.add_argument("--backbone-name", default="resnet18", help="MONAI 3D ResNet backbone name.")
    parser.add_argument(
        "--embedding-pool",
        choices=("avg", "avg_max"),
        default="avg",
        help="How to pool the penultimate 3D feature map.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=1, help="Batch size for embedding extraction.")
    parser.add_argument("--target-depth", type=int, default=64, help="Final crop depth.")
    parser.add_argument("--target-height", type=int, default=96, help="Final crop height.")
    parser.add_argument("--target-width", type=int, default=96, help="Final crop width.")
    parser.add_argument("--hu-min", type=int, default=-1000, help="Lower HU clipping bound.")
    parser.add_argument("--hu-max", type=int, default=400, help="Upper HU clipping bound.")
    parser.add_argument("--bbox-margin", type=int, default=8, help="Voxel margin around the tumor bounding box.")
    parser.add_argument("--spacing-z", type=float, default=2.5, help="Target spacing along z.")
    parser.add_argument("--spacing-y", type=float, default=1.0, help="Target spacing along y.")
    parser.add_argument("--spacing-x", type=float, default=1.0, help="Target spacing along x.")
    parser.add_argument("--skip-fit", action="store_true", help="Stop after deep feature extraction and table preparation.")
    parser.add_argument("--skip-roc", action="store_true", help="Do not render ROC PNGs after endpoint fitting.")
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional limit for a quick Colab smoke test. Uses the first N manifest rows.",
    )
    return parser


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


def maybe_limit_manifest(manifest_path: Path, output_dir: Path, max_patients: int | None) -> Path:
    if max_patients is None:
        return manifest_path.resolve()

    records = read_manifest(manifest_path)
    if max_patients < 1:
        raise ValueError("max_patients must be at least 1.")
    subset_records = records[:max_patients]
    subset_path = output_dir / f"manifest_first_{max_patients}.csv"
    return write_subset_manifest(subset_records, subset_path)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = maybe_limit_manifest(Path(args.manifest).resolve(), output_dir, args.max_patients)
    generated_config_path = (
        Path(args.generated_config).resolve()
        if args.generated_config
        else (output_dir / "colab_pipeline.generated.json").resolve()
    )

    return PipelineConfig(
        config_path=generated_config_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        checkpoint_filename="resnet18_backbone.pt",
        preprocessing=PreprocessingConfig(
            target_depth=args.target_depth,
            target_height=args.target_height,
            target_width=args.target_width,
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
            weights_path=Path(args.weights_path).resolve(),
        ),
        features=FeatureConfig(
            batch_size=args.feature_batch_size,
            output_filename="deep_features.csv",
        ),
        tabular=TabularConfig(
            xlsx_path=Path(args.xlsx_path).resolve(),
            prepared_output_filename="prepared_dataset.csv",
            evaluation_output_filename="endpoint_metrics.json",
            predictions_output_filename="endpoint_predictions.csv",
            radiomics_aggregations=("mean", "max"),
            endpoint_windows_months=(12, 36),
        ),
    )


def write_generated_config(config: PipelineConfig) -> Path:
    payload = {
        "manifest_path": str(config.manifest_path),
        "output_dir": str(config.output_dir),
        "checkpoint_filename": config.checkpoint_filename,
        "preprocessing": config.preprocessing.__dict__,
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
            "xlsx_path": str(config.tabular.xlsx_path) if config.tabular.xlsx_path else None,
            "prepared_output_filename": config.tabular.prepared_output_filename,
            "evaluation_output_filename": config.tabular.evaluation_output_filename,
            "predictions_output_filename": config.tabular.predictions_output_filename,
            "radiomics_aggregations": list(config.tabular.radiomics_aggregations),
            "endpoint_windows_months": list(config.tabular.endpoint_windows_months),
        },
    }
    config.config_path.parent.mkdir(parents=True, exist_ok=True)
    with config.config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return config.config_path


def validate_inputs(config: PipelineConfig) -> None:
    required_paths = {
        "manifest": config.manifest_path,
        "xlsx": config.tabular.xlsx_path,
        "weights": config.model.weights_path,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)
    validate_inputs(config)
    generated_config_path = write_generated_config(config)
    print(f"Generated config: {generated_config_path}")

    checkpoint_path = prepare_backbone_checkpoint(config)
    feature_path = extract_patient_embeddings(config, checkpoint_path=checkpoint_path)
    prepared_path = prepare_tabular_dataset(config)

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Deep features saved to: {feature_path}")
    print(f"Prepared dataset saved to: {prepared_path}")

    if args.skip_fit:
        print("Skipping endpoint fitting because --skip-fit was provided.")
        return

    metrics_path, predictions_path = fit_endpoint_models(config, prepared_table_path=prepared_path)
    print(f"Endpoint metrics saved to: {metrics_path}")
    print(f"Endpoint predictions saved to: {predictions_path}")

    if args.skip_roc:
        print("Skipping ROC plotting because --skip-roc was provided.")
        return

    roc_paths = plot_endpoint_roc_curves(
        predictions_path=config.prediction_output_path,
        output_dir=config.output_dir / "roc_curves",
        include_base_models=True,
    )
    print("ROC plots:")
    for roc_path in roc_paths:
        print(f"  - {roc_path}")


if __name__ == "__main__":
    main()
