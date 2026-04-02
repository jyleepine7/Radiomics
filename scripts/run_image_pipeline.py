from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nsclc_unet.config import load_config
from nsclc_unet.tabular import prepare_tabular_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CT deep features with a MONAI 3D ResNet and fit endpoint models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Prepare and snapshot the MONAI 3D ResNet backbone checkpoint.",
    )
    train_parser.add_argument("--config", required=True, help="Path to the JSON config.")

    extract_parser = subparsers.add_parser("extract", help="Extract patient-level 3D deep features.")
    extract_parser.add_argument("--config", required=True, help="Path to the JSON config.")
    extract_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override. If omitted, output_dir/checkpoint_filename is used when available.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-tabular",
        help="Build a patient-level dataset from Radiomics_Clinical.xlsx and optional deep features.",
    )
    prepare_parser.add_argument("--config", required=True, help="Path to the JSON config.")
    prepare_parser.add_argument(
        "--xlsx-path",
        default=None,
        help="Optional XLSX override. Defaults to tabular.xlsx_path in config.",
    )
    prepare_parser.add_argument(
        "--deep-features-path",
        default=None,
        help="Optional deep features CSV override. Defaults to output_dir/deep_features.csv if present.",
    )

    fit_parser = subparsers.add_parser(
        "fit-endpoints",
        help="Run nested CV endpoint modeling on the prepared patient-level table.",
    )
    fit_parser.add_argument("--config", required=True, help="Path to the JSON config.")
    fit_parser.add_argument(
        "--table-path",
        default=None,
        help="Optional prepared CSV override. Defaults to output_dir/prepared_dataset.csv.",
    )

    plot_parser = subparsers.add_parser(
        "plot-roc",
        help="Plot ROC curve PNG files from endpoint_predictions.csv.",
    )
    plot_parser.add_argument("--config", required=True, help="Path to the JSON config.")
    plot_parser.add_argument(
        "--predictions-path",
        default=None,
        help="Optional endpoint predictions CSV override. Defaults to output_dir/endpoint_predictions.csv.",
    )
    plot_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override. Defaults to output_dir/roc_curves.",
    )
    plot_parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Only plot the ensemble ROC curve instead of all base learners.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))

    if args.command == "train":
        from nsclc_unet.train import prepare_backbone_checkpoint

        prepare_backbone_checkpoint(config)
    elif args.command == "extract":
        from nsclc_unet.features import extract_patient_embeddings

        extract_patient_embeddings(config, checkpoint_path=Path(args.checkpoint) if args.checkpoint else None)
    elif args.command == "prepare-tabular":
        prepare_tabular_dataset(
            config,
            xlsx_path=Path(args.xlsx_path) if args.xlsx_path else None,
            deep_features_path=Path(args.deep_features_path) if args.deep_features_path else None,
        )
    elif args.command == "fit-endpoints":
        from nsclc_unet.modeling import fit_endpoint_models

        fit_endpoint_models(config, prepared_table_path=Path(args.table_path) if args.table_path else None)
    elif args.command == "plot-roc":
        from nsclc_unet.plotting import plot_endpoint_roc_curves

        predictions_path = Path(args.predictions_path) if args.predictions_path else config.prediction_output_path
        output_dir = Path(args.output_dir) if args.output_dir else (config.output_dir / "roc_curves")
        plot_endpoint_roc_curves(
            predictions_path=predictions_path,
            output_dir=output_dir,
            include_base_models=not args.ensemble_only,
        )
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
