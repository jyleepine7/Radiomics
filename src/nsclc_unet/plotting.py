from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def plot_endpoint_roc_curves(
    predictions_path: Path,
    output_dir: Path,
    include_base_models: bool = True,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to plot ROC curves. Install it with `pip install matplotlib`."
        ) from exc

    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {predictions_path}")

    prediction_frame = pd.read_csv(predictions_path)
    required_columns = {"endpoint", "label", "prob_ensemble"}
    missing = required_columns.difference(prediction_frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_probability_columns = ["prob_ensemble"]
    if include_base_models:
        candidate_probability_columns.extend(
            sorted(column for column in prediction_frame.columns if column.startswith("prob_") and column != "prob_ensemble")
        )

    created_paths: list[Path] = []
    for endpoint_name, endpoint_frame in prediction_frame.groupby("endpoint"):
        y_true = endpoint_frame["label"].astype(int).to_numpy()
        if len(set(y_true)) < 2:
            print(f"Skipping ROC plot for {endpoint_name}: need both positive and negative labels.")
            continue

        figure, axis = plt.subplots(figsize=(7, 6))
        plotted_any = False
        for probability_column in candidate_probability_columns:
            if probability_column not in endpoint_frame.columns:
                continue
            scores = endpoint_frame[probability_column].astype(float)
            if scores.isna().all():
                continue

            valid_mask = scores.notna().to_numpy()
            y_valid = y_true[valid_mask]
            scores_valid = scores.to_numpy(dtype=float)[valid_mask]
            if len(set(y_valid)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_valid, scores_valid)
            auc = roc_auc_score(y_valid, scores_valid)
            label = probability_column.replace("prob_", "")
            axis.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.3f})")
            plotted_any = True

        if not plotted_any:
            plt.close(figure)
            print(f"Skipping ROC plot for {endpoint_name}: no usable probability columns found.")
            continue

        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title(f"ROC Curves - {endpoint_name}")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.25)
        figure.tight_layout()

        output_path = output_dir / f"roc_{endpoint_name}.png"
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        created_paths.append(output_path)
        print(f"Saved ROC curve to {output_path}")

    if not created_paths:
        raise ValueError("No ROC curves were generated from the provided prediction file.")

    return created_paths
