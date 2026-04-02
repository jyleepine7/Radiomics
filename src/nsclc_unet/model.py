from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsclc_unet.config import ModelConfig


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_monai() -> object:
    try:
        from monai.networks import nets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MONAI is required for the 3D ResNet feature extractor. Install it with `pip install monai`."
        ) from exc
    return nets


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for candidate_key in ("model_state_dict", "state_dict", "net", "network_state_dict"):
            candidate = payload.get(candidate_key)
            if isinstance(candidate, dict):
                return candidate
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format for external weights.")


def _strip_prefix_if_present(key: str, prefix: str) -> str:
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def _sanitize_external_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    sanitized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in state_dict.items():
        key = raw_key
        for prefix in ("module.", "backbone.", "model.", "resnet."):
            key = _strip_prefix_if_present(key, prefix)

        if key.startswith("fc.") or key.startswith("classifier.") or key.startswith("head."):
            continue

        target_tensor = model_state_dict.get(key)
        if target_tensor is None:
            continue
        if target_tensor.shape != tensor.shape:
            continue
        sanitized[key] = tensor
    return sanitized


class MONAIResNetFeatureExtractor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_pool = config.embedding_pool
        self.backbone = self._build_backbone(config)
        if config.weights_path is not None:
            self.load_external_weights(config.weights_path)

    def _build_backbone(self, config: ModelConfig) -> nn.Module:
        nets = _require_monai()
        factories = {
            "resnet10": getattr(nets, "resnet10", None),
            "resnet18": getattr(nets, "resnet18", None),
            "resnet34": getattr(nets, "resnet34", None),
            "resnet50": getattr(nets, "resnet50", None),
        }
        factory = factories.get(config.backbone_name)
        if factory is None:
            raise ValueError(
                f"Unsupported MONAI backbone '{config.backbone_name}'. "
                f"Choose one of: {sorted(name for name, fn in factories.items() if fn is not None)}"
            )

        return factory(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=config.in_channels,
            conv1_t_size=config.conv1_t_size,
            conv1_t_stride=config.conv1_t_stride,
            no_max_pool=config.no_max_pool,
            shortcut_type=config.shortcut_type,
            widen_factor=config.widen_factor,
            feed_forward=False,
            bias_downsample=config.bias_downsample,
        )

    def load_external_weights(self, weights_path: Path) -> None:
        resolved = weights_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"External weights not found: {resolved}")

        payload = torch.load(resolved, map_location="cpu")
        external_state_dict = _extract_state_dict(payload)
        current_state_dict = self.backbone.state_dict()
        filtered_state_dict = _sanitize_external_state_dict(external_state_dict, current_state_dict)
        if not filtered_state_dict:
            raise ValueError(
                f"No compatible backbone weights were found in {resolved}. "
                "Check that the checkpoint matches the selected MONAI ResNet architecture."
            )
        missing, unexpected = self.backbone.load_state_dict(filtered_state_dict, strict=False)
        print(
            f"Loaded {len(filtered_state_dict)} backbone tensors from {resolved}. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def _forward_backbone_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        if hasattr(model, "relu"):
            x = model.relu(x)
        elif hasattr(model, "act"):
            x = model.act(x)
        if not getattr(model, "no_max_pool", False) and hasattr(model, "maxpool"):
            x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self._forward_backbone_feature_map(x)
        avg_pool = F.adaptive_avg_pool3d(feature_map, output_size=1).flatten(start_dim=1)
        if self.embedding_pool == "avg_max":
            max_pool = F.adaptive_max_pool3d(feature_map, output_size=1).flatten(start_dim=1)
            return torch.cat([avg_pool, max_pool], dim=1)
        return avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_embedding(x)


def build_feature_extractor(
    config: ModelConfig,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> MONAIResNetFeatureExtractor:
    model = MONAIResNetFeatureExtractor(config)
    checkpoint = checkpoint_path.resolve() if checkpoint_path is not None else None
    if checkpoint is not None and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(payload)
        model.load_state_dict(state_dict, strict=False)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
