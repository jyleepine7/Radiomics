from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    patient_id: str
    image_path: Path
    mask_path: Path


def read_manifest(path: Path) -> list[ManifestRecord]:
    base_dir = path.resolve().parent
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "image_path", "mask_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for row in reader:
            patient_id = (row.get("patient_id") or "").strip()
            image_raw = (row.get("image_path") or "").strip()
            mask_raw = (row.get("mask_path") or "").strip()
            if not patient_id or not image_raw or not mask_raw:
                continue

            image_path = Path(image_raw)
            mask_path = Path(mask_raw)
            if not image_path.is_absolute():
                image_path = (base_dir / image_path).resolve()
            if not mask_path.is_absolute():
                mask_path = (base_dir / mask_path).resolve()

            records.append(ManifestRecord(patient_id=patient_id, image_path=image_path, mask_path=mask_path))

    if not records:
        raise ValueError(f"No valid rows found in manifest: {path}")
    return records

