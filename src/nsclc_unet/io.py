from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PNG_LIKE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nibabel is required to load NIfTI files.") from exc

    image = nib.load(str(path))
    return np.asarray(image.get_fdata(), dtype=np.float32)


def _load_numpy(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)

    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        first_key = loaded.files[0]
        return loaded[first_key].astype(np.float32)
    return np.asarray(loaded, dtype=np.float32)


def _load_dicom_with_simpleitk(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SimpleITK is not available.") from exc

    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(path))
    if not file_names:
        raise ValueError(f"No DICOM slices found in directory: {path}")
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image).astype(np.float32)


def _load_dicom_with_pydicom(path: Path) -> np.ndarray:
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pydicom is not available.") from exc

    dicom_files = [file for file in path.iterdir() if file.is_file()]
    if not dicom_files:
        raise ValueError(f"No files found in DICOM directory: {path}")

    slices = []
    for file in dicom_files:
        try:
            dataset = pydicom.dcmread(str(file), force=True)
        except Exception:
            continue
        if not hasattr(dataset, "PixelData"):
            continue
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixel_array = dataset.pixel_array.astype(np.float32) * slope + intercept
        z_position = None
        if hasattr(dataset, "ImagePositionPatient"):
            try:
                z_position = float(dataset.ImagePositionPatient[2])
            except Exception:
                z_position = None
        instance_number = int(getattr(dataset, "InstanceNumber", 0))
        slices.append((z_position if z_position is not None else instance_number, pixel_array))

    if not slices:
        raise ValueError(f"No readable DICOM slices found in directory: {path}")

    slices.sort(key=lambda item: item[0])
    return np.stack([pixel_array for _, pixel_array in slices], axis=0).astype(np.float32)


def _read_png_file(file: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required to load PNG stacks.") from exc

    with Image.open(file) as image:
        array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        # If a grayscale PNG was saved with redundant channels, keep one channel.
        array = array[..., 0]
    return array.astype(np.float32)


def _load_png_stack(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required to load PNG stacks.") from exc

    image_files = sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in PNG_LIKE_SUFFIXES)
    if not image_files:
        raise ValueError(f"No PNG/JPG/TIFF/BMP slices found in directory: {path}")

    slices = [_read_png_file(file) for file in image_files]
    return np.stack(slices, axis=0).astype(np.float32)


def load_volume(path: Path) -> np.ndarray:
    resolved = path.resolve()

    if resolved.is_dir():
        if any(file.is_file() and file.suffix.lower() in PNG_LIKE_SUFFIXES for file in resolved.iterdir()):
            return _load_png_stack(resolved)
        try:
            return _load_dicom_with_simpleitk(resolved)
        except ModuleNotFoundError:
            return _load_dicom_with_pydicom(resolved)

    if resolved.name.endswith(".nii.gz") or resolved.suffix == ".nii":
        return _load_nifti(resolved)

    if resolved.suffix in {".npy", ".npz"}:
        return _load_numpy(resolved)

    raise ValueError(f"Unsupported image format: {resolved}")


def load_case(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_path = image_path.resolve()
    mask_path = mask_path.resolve()

    if image_path.is_dir() and mask_path.is_dir():
        image_files = {
            file.name: file
            for file in image_path.iterdir()
            if file.is_file() and file.suffix.lower() in PNG_LIKE_SUFFIXES
        }
        mask_files = {
            file.name: file
            for file in mask_path.iterdir()
            if file.is_file() and file.suffix.lower() in PNG_LIKE_SUFFIXES
        }
        if image_files and mask_files:
            shared_names = sorted(set(image_files).intersection(mask_files))
            if not shared_names:
                raise ValueError(f"No overlapping PNG slice names between {image_path} and {mask_path}")
            image = np.stack([_read_png_file(image_files[name]) for name in shared_names], axis=0).astype(np.float32)
            mask = np.stack([_read_png_file(mask_files[name]) for name in shared_names], axis=0).astype(np.float32)
        else:
            image = load_volume(image_path).astype(np.float32)
            mask = load_volume(mask_path).astype(np.float32)
    else:
        image = load_volume(image_path).astype(np.float32)
        mask = load_volume(mask_path).astype(np.float32)

    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask shape mismatch for case {image_path.name}: {image.shape} vs {mask.shape}"
        )

    mask = (mask > 0).astype(np.float32)
    return image, mask


def save_feature_rows(rows: list[dict[str, str | float]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No feature rows were generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
