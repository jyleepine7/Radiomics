from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    target_height: int = 256
    target_width: int = 256
    hu_min: int = -1000
    hu_max: int = 400
    min_mask_pixels: int = 8
    bbox_margin: int = 16

    @property
    def target_size(self) -> tuple[int, int]:
        return (self.target_height, self.target_width)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    random_seed: int = 42
    num_workers: int = 0
    early_stopping_patience: int = 8


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 16
    dropout: float = 0.1
    embedding_pool: str = "avg"


@dataclass(frozen=True)
class FeatureConfig:
    batch_size: int = 16
    aggregate_mean: bool = True
    aggregate_max: bool = True
    output_filename: str = "deep_features.csv"


@dataclass(frozen=True)
class TabularConfig:
    xlsx_path: Path | None = None
    prepared_output_filename: str = "prepared_dataset.csv"
    evaluation_output_filename: str = "endpoint_metrics.json"
    predictions_output_filename: str = "endpoint_predictions.csv"
    radiomics_aggregations: tuple[str, ...] = ("mean", "max")
    endpoint_windows_months: tuple[int, ...] = (12, 36)


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    manifest_path: Path
    output_dir: Path
    checkpoint_filename: str
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: ModelConfig
    features: FeatureConfig
    tabular: TabularConfig

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_filename

    @property
    def feature_output_path(self) -> Path:
        return self.output_dir / self.features.output_filename

    @property
    def prepared_dataset_path(self) -> Path:
        return self.output_dir / self.tabular.prepared_output_filename

    @property
    def evaluation_output_path(self) -> Path:
        return self.output_dir / self.tabular.evaluation_output_filename

    @property
    def prediction_output_path(self) -> Path:
        return self.output_dir / self.tabular.predictions_output_filename


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path: Path) -> PipelineConfig:
    resolved_path = path.resolve()
    base_dir = resolved_path.parent
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    preprocessing = PreprocessingConfig(**raw.get("preprocessing", {}))
    training = TrainingConfig(**raw.get("training", {}))
    model = ModelConfig(**raw.get("model", {}))
    features = FeatureConfig(**raw.get("features", {}))
    raw_tabular = dict(raw.get("tabular", {}))
    xlsx_path_raw = raw_tabular.pop("xlsx_path", None)
    tabular = TabularConfig(
        xlsx_path=_resolve_path(base_dir, xlsx_path_raw) if xlsx_path_raw else None,
        **raw_tabular,
    )

    output_dir = _resolve_path(base_dir, raw.get("output_dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return PipelineConfig(
        config_path=resolved_path,
        manifest_path=_resolve_path(base_dir, raw["manifest_path"]),
        output_dir=output_dir,
        checkpoint_filename=raw.get("checkpoint_filename", "unet_best.pt"),
        preprocessing=preprocessing,
        training=training,
        model=model,
        features=features,
        tabular=tabular,
    )
