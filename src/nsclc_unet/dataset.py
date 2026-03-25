from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from nsclc_unet.config import PreprocessingConfig
from nsclc_unet.io import load_case
from nsclc_unet.manifest import ManifestRecord
from nsclc_unet.preprocess import extract_tumor_slices


@dataclass
class SliceSample:
    patient_id: str
    image: np.ndarray
    mask: np.ndarray


class CTSliceDataset(Dataset):
    def __init__(
        self,
        records: list[ManifestRecord],
        preprocessing: PreprocessingConfig,
        augment: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.records = records
        self.preprocessing = preprocessing
        self.augment = augment
        self.rng = np.random.default_rng(random_seed)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[SliceSample]:
        samples: list[SliceSample] = []
        for record in self.records:
            image, mask = load_case(record.image_path, record.mask_path)
            for image_slice, mask_slice in extract_tumor_slices(image, mask, self.preprocessing):
                samples.append(SliceSample(patient_id=record.patient_id, image=image_slice, mask=mask_slice))

        if not samples:
            raise ValueError("No slice samples were created. Check the manifest and masks.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = sample.image
        mask = sample.mask
        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image[None, ...]).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        return image_tensor, mask_tensor

