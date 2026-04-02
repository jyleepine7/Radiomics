from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from nsclc_unet.config import PreprocessingConfig
from nsclc_unet.io import load_case_with_metadata
from nsclc_unet.manifest import ManifestRecord
from nsclc_unet.preprocess import prepare_tumor_volume


@dataclass
class VolumeSample:
    patient_id: str
    image: np.ndarray
    mask: np.ndarray


class CTVolumeDataset(Dataset):
    def __init__(
        self,
        records: list[ManifestRecord],
        preprocessing: PreprocessingConfig,
    ) -> None:
        self.records = records
        self.preprocessing = preprocessing
        self.samples = self._build_samples()

    def _build_samples(self) -> list[VolumeSample]:
        samples: list[VolumeSample] = []
        for record in self.records:
            image, mask, spacing = load_case_with_metadata(record.image_path, record.mask_path)
            volume, volume_mask = prepare_tumor_volume(image, mask, self.preprocessing, spacing=spacing)
            samples.append(VolumeSample(patient_id=record.patient_id, image=volume, mask=volume_mask))

        if not samples:
            raise ValueError("No volume samples were created. Check the manifest and masks.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_tensor = torch.from_numpy(sample.image[None, ...]).float()
        mask_tensor = torch.from_numpy(sample.mask[None, ...]).float()
        return image_tensor, mask_tensor


CTSliceDataset = CTVolumeDataset
SliceSample = VolumeSample
