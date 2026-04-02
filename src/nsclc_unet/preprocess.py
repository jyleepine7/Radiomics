from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from nsclc_unet.config import PreprocessingConfig


def clamp_and_normalize_hu(volume: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / float(hu_max - hu_min)
    return normalized.astype(np.float32)


def compute_3d_bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        return (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))

    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(mask.shape))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs))


def crop_to_bbox(image: np.ndarray, mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    bbox = compute_3d_bbox(mask, margin)
    return image[bbox].astype(np.float32), mask[bbox].astype(np.float32)


def resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(volume[None, None]).float()
    align_corners = False if mode in {"trilinear", "bilinear"} else None
    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode=mode,
        align_corners=align_corners,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    mode: str,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume for resampling, got shape {volume.shape}")

    resampled_shape = []
    for size, src_spacing, dst_spacing in zip(volume.shape, source_spacing, target_spacing):
        scaled = max(1, int(round(float(size) * float(src_spacing) / float(dst_spacing))))
        resampled_shape.append(scaled)
    return resize_volume(volume, tuple(resampled_shape), mode=mode)


def prepare_tumor_volume(
    image: np.ndarray,
    mask: np.ndarray,
    config: PreprocessingConfig,
    spacing: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image = clamp_and_normalize_hu(image, config.hu_min, config.hu_max)
    image, mask = crop_to_bbox(image, mask, config.bbox_margin)

    target_spacing = config.target_spacing
    if spacing is not None and target_spacing is not None:
        image = resample_volume(image, spacing, target_spacing, mode="trilinear")
        mask = resample_volume(mask, spacing, target_spacing, mode="nearest")

    image = resize_volume(image, config.target_shape, mode="trilinear")
    mask = resize_volume(mask, config.target_shape, mode="nearest")
    mask = (mask > 0.5).astype(np.float32)
    return image.astype(np.float32), mask


def resize_slice(image_slice: np.ndarray, mask_slice: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = torch.from_numpy(image_slice[None, None]).float()
    mask_tensor = torch.from_numpy(mask_slice[None, None]).float()
    resized_image = F.interpolate(image_tensor, size=target_size, mode="bilinear", align_corners=False)
    resized_mask = F.interpolate(mask_tensor, size=target_size, mode="nearest")
    return resized_image.squeeze(0).squeeze(0).numpy().astype(np.float32), resized_mask.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _select_slice_indices(mask: np.ndarray) -> list[int]:
    slice_areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    positive = np.flatnonzero(slice_areas > 0).tolist()
    if positive:
        return positive
    fallback_index = int(slice_areas.argmax()) if slice_areas.size else 0
    return [fallback_index]


def extract_tumor_slices(image: np.ndarray, mask: np.ndarray, config: PreprocessingConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    volume, volume_mask = prepare_tumor_volume(image, mask, config, spacing=None)
    slices: list[tuple[np.ndarray, np.ndarray]] = []
    for slice_index in _select_slice_indices(volume_mask):
        slices.append((volume[slice_index], volume_mask[slice_index]))
    return slices


def patient_level_slice_count(mask: np.ndarray, min_mask_pixels: int = 1) -> int:
    _ = min_mask_pixels
    return len(_select_slice_indices(mask))
