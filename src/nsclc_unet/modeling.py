from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from nsclc_unet.config import PipelineConfig


class DropAllMissingColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing within the current training fold."""

    def fit(self, X, y=None):
        array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.keep_mask_ = ~pd.isna(array).all(axis=0)
        if not np.any(self.keep_mask_):
            self.keep_mask_ = np.ones(array.shape[1], dtype=bool)
        return self

    def transform(self, X):
        array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array[:, self.keep_mask_]


def _choose_n_splits(y: pd.Series, desired: int) -> int:
    value_counts = y.value_counts()
    if value_counts.empty:
        raise ValueError("No labels available for cross-validation.")
    min_class_count = int(value_counts.min())
    return max(2, min(desired, min_class_count, len(y)))


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("label_", "eligible_", "time_", "event_")
    return [
        column
        for column in frame.columns
        if column != "patient_id" and not column.startswith(excluded_prefixes)
    ]


def _build_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        [
            ("drop_empty", DropAllMissingColumns()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("drop_empty", DropAllMissingColumns()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.0,
    )


def _model_specs(random_seed: int) -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=4000, class_weight="balanced"),
            {"model__C": [0.1, 1.0, 5.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_seed, class_weight="balanced"),
            {"model__n_estimators": [200, 500], "model__max_depth": [None, 4, 8]},
        ),
        "svm": (
            SVC(probability=True, class_weight="balanced", random_state=random_seed),
            {"model__C": [0.5, 1.0, 5.0], "model__kernel": ["rbf", "linear"]},
        ),
        "gbm": (
            GradientBoostingClassifier(random_state=random_seed),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1]},
        ),
        "mlp": (
            MLPClassifier(
                random_state=random_seed,
                max_iter=800,
                early_stopping=True,
                hidden_layer_sizes=(64, 32),
            ),
            {"model__alpha": [1e-4, 1e-3], "model__hidden_layer_sizes": [(64, 32), (128, 64)]},
        ),
    }


def _prepare_endpoint_frame(table: pd.DataFrame, endpoint_name: str) -> pd.DataFrame:
    eligible_column = endpoint_name.replace("label_", "eligible_")
    filtered = table[table[eligible_column] == 1].copy()
    filtered = filtered[filtered[endpoint_name].notna()].copy()
    filtered[endpoint_name] = filtered[endpoint_name].astype(int)
    return filtered.reset_index(drop=True)


def _nested_cv_predictions(frame: pd.DataFrame, endpoint_name: str, random_seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    y = frame[endpoint_name].astype(int)
    feature_columns = _feature_columns(frame)
    X = frame[feature_columns]

    outer_splits = _choose_n_splits(y, desired=5)
    inner_splits = _choose_n_splits(y, desired=3)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_seed)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed)

    base_specs = _model_specs(random_seed)
    ensemble_predictions = np.zeros(len(frame), dtype=np.float64)
    prediction_frame = pd.DataFrame({"patient_id": frame["patient_id"], "endpoint": endpoint_name})

    for model_name in base_specs:
        prediction_frame[f"prob_{model_name}"] = np.nan
    prediction_frame["prob_ensemble"] = np.nan
    prediction_frame["label"] = y.values

    selected_hyperparameters: list[dict[str, object]] = []

    for fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        fold_probabilities: list[np.ndarray] = []
        for model_name, (model, parameter_grid) in base_specs.items():
            pipeline = Pipeline(
                [
                    ("preprocessor", _build_preprocessor(X_train, feature_columns)),
                    ("model", model),
                ]
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            probabilities = search.predict_proba(X_test)[:, 1]
            prediction_frame.loc[test_idx, f"prob_{model_name}"] = probabilities
            fold_probabilities.append(probabilities)
            selected_hyperparameters.append(
                {
                    "endpoint": endpoint_name,
                    "outer_fold": fold_index,
                    "model": model_name,
                    "best_params": search.best_params_,
                    "best_inner_score": float(search.best_score_),
                }
            )

        averaged = np.mean(np.column_stack(fold_probabilities), axis=1)
        ensemble_predictions[test_idx] = averaged
        prediction_frame.loc[test_idx, "prob_ensemble"] = averaged

    metrics: dict[str, object] = {
        "endpoint": endpoint_name,
        "n_samples": int(len(frame)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "outer_folds": outer_splits,
        "inner_folds": inner_splits,
        "auroc_ensemble": float(roc_auc_score(y, ensemble_predictions)),
        "selected_hyperparameters": selected_hyperparameters,
    }

    for model_name in base_specs:
        model_probabilities = prediction_frame[f"prob_{model_name}"].to_numpy(dtype=float)
        metrics[f"auroc_{model_name}"] = float(roc_auc_score(y, model_probabilities))

    return metrics, prediction_frame


def fit_endpoint_models(config: PipelineConfig, prepared_table_path: Path | None = None) -> tuple[Path, Path]:
    table_path = prepared_table_path or config.prepared_dataset_path
    if not table_path.exists():
        raise FileNotFoundError(f"Prepared table not found: {table_path}")

    table = pd.read_csv(table_path)
    endpoint_columns = [column for column in table.columns if column.startswith("label_")]
    if not endpoint_columns:
        raise ValueError("No endpoint label columns were found in the prepared dataset.")

    all_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for endpoint_name in endpoint_columns:
        endpoint_frame = _prepare_endpoint_frame(table, endpoint_name)
        if endpoint_frame.empty or endpoint_frame[endpoint_name].nunique() < 2:
            print(f"Skipping {endpoint_name}: not enough labeled samples with both classes.")
            continue
        metrics, predictions = _nested_cv_predictions(
            endpoint_frame,
            endpoint_name=endpoint_name,
            random_seed=config.training.random_seed,
        )
        all_metrics.append(metrics)
        prediction_frames.append(predictions)
        print(
            f"{endpoint_name}: AUROC ensemble={metrics['auroc_ensemble']:.3f} "
            f"(n={metrics['n_samples']}, pos={metrics['n_positive']})"
        )

    if not all_metrics:
        raise ValueError("No endpoint models were fit successfully.")

    with config.evaluation_output_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

    prediction_output = pd.concat(prediction_frames, ignore_index=True)
    prediction_output.to_csv(config.prediction_output_path, index=False)
    print(f"Saved endpoint metrics to {config.evaluation_output_path}")
    print(f"Saved endpoint predictions to {config.prediction_output_path}")
    return config.evaluation_output_path, config.prediction_output_path
