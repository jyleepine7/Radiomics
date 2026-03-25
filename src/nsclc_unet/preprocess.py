from __future__ import annotations

from typing import Iterable

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


def resize_slice(image_slice: np.ndarray, mask_slice: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = torch.from_numpy(image_slice[None, None]).float()
    mask_tensor = torch.from_numpy(mask_slice[None, None]).float()

    resized_image = F.interpolate(image_tensor, size=target_size, mode="bilinear", align_corners=False)
    resized_mask = F.interpolate(mask_tensor, size=target_size, mode="nearest")

    return resized_image.squeeze(0).squeeze(0).numpy().astype(np.float32), resized_mask.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _select_slice_indices(mask: np.ndarray, min_mask_pixels: int) -> list[int]:
    slice_areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    positive = np.flatnonzero(slice_areas >= min_mask_pixels).tolist()
    if positive:
        return positive
    fallback_index = int(slice_areas.argmax()) if slice_areas.size else 0
    return [fallback_index]


def extract_tumor_slices(image: np.ndarray, mask: np.ndarray, config: PreprocessingConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    image = clamp_and_normalize_hu(image, config.hu_min, config.hu_max)
    bbox = compute_3d_bbox(mask, config.bbox_margin)
    image = image[bbox]
    mask = mask[bbox]

    slices: list[tuple[np.ndarray, np.ndarray]] = []
    for slice_index in _select_slice_indices(mask, config.min_mask_pixels):
        image_slice = image[slice_index]
        mask_slice = mask[slice_index]
        resized_image, resized_mask = resize_slice(image_slice, mask_slice, config.target_size)
        slices.append((resized_image, resized_mask))
    return slices


def patient_level_slice_count(mask: np.ndarray, min_mask_pixels: int) -> int:
    return len(_select_slice_indices(mask, min_mask_pixels))

