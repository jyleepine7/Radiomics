from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    target_depth: int = 64
    target_height: int = 96
    target_width: int = 96
    hu_min: int = -1000
    hu_max: int = 400
    bbox_margin: int = 8
    target_spacing_z: float | None = 2.5
    target_spacing_y: float | None = 1.0
    target_spacing_x: float | None = 1.0

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (self.target_depth, self.target_height, self.target_width)

    @property
    def target_spacing(self) -> tuple[float, float, float] | None:
        values = (self.target_spacing_z, self.target_spacing_y, self.target_spacing_x)
        if any(value is None for value in values):
            return None
        return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    random_seed: int = 42
    num_workers: int = 0
    early_stopping_patience: int = 3


@dataclass(frozen=True)
class ModelConfig:
    backbone_name: str = "resnet18"
    in_channels: int = 1
    embedding_pool: str = "avg"
    conv1_t_size: int = 7
    conv1_t_stride: int = 1
    no_max_pool: bool = False
    shortcut_type: str = "A"
    widen_factor: float = 1.0
    bias_downsample: bool = True
    weights_path: Path | None = None


@dataclass(frozen=True)
class FeatureConfig:
    batch_size: int = 1
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

    raw_model = dict(raw.get("model", {}))
    weights_path_raw = raw_model.pop("weights_path", None)
    model = ModelConfig(
        weights_path=_resolve_path(base_dir, weights_path_raw) if weights_path_raw else None,
        **raw_model,
    )

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
        checkpoint_filename=raw.get("checkpoint_filename", "resnet18_backbone.pt"),
        preprocessing=preprocessing,
        training=training,
        model=model,
        features=features,
        tabular=tabular,
    )
