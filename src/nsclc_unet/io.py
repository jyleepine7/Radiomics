from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PNG_LIKE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class VolumeData:
    array: np.ndarray
    spacing: tuple[float, float, float] | None = None


def _ensure_3d(array: np.ndarray, source: Path) -> np.ndarray:
    squeezed = np.asarray(array, dtype=np.float32).squeeze()
    if squeezed.ndim != 3:
        raise ValueError(f"Expected a 3D volume from {source}, got shape {squeezed.shape}")
    return squeezed.astype(np.float32)


def _load_nifti(path: Path) -> VolumeData:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nibabel is required to load NIfTI files.") from exc

    image = nib.load(str(path))
    array_xyz = _ensure_3d(np.asarray(image.get_fdata(), dtype=np.float32), path)
    spacing_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
    array_zyx = np.transpose(array_xyz, (2, 1, 0)).astype(np.float32)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return VolumeData(array=array_zyx, spacing=spacing_zyx)


def _load_numpy(path: Path) -> VolumeData:
    if path.suffix == ".npy":
        array = np.load(path)
    else:
        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            first_key = loaded.files[0]
            array = loaded[first_key]
        else:
            array = loaded
    return VolumeData(array=_ensure_3d(array, path), spacing=None)


def _load_dicom_with_simpleitk(path: Path) -> VolumeData:
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
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return VolumeData(array=_ensure_3d(array, path), spacing=spacing_zyx)


def _load_dicom_with_pydicom(path: Path) -> VolumeData:
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pydicom is not available.") from exc

    dicom_files = [file for file in path.iterdir() if file.is_file()]
    if not dicom_files:
        raise ValueError(f"No files found in DICOM directory: {path}")

    slices: list[tuple[float, np.ndarray]] = []
    z_positions: list[float] = []
    pixel_spacing: tuple[float, float] | None = None
    fallback_thickness: float | None = None
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
        instance_number = float(getattr(dataset, "InstanceNumber", 0))
        sort_key = z_position if z_position is not None else instance_number
        if z_position is not None:
            z_positions.append(z_position)
        if hasattr(dataset, "PixelSpacing"):
            try:
                pixel_spacing = (float(dataset.PixelSpacing[0]), float(dataset.PixelSpacing[1]))
            except Exception:
                pixel_spacing = pixel_spacing
        if hasattr(dataset, "SliceThickness"):
            try:
                fallback_thickness = float(dataset.SliceThickness)
            except Exception:
                fallback_thickness = fallback_thickness
        slices.append((sort_key, pixel_array))

    if not slices:
        raise ValueError(f"No readable DICOM slices found in directory: {path}")

    slices.sort(key=lambda item: item[0])
    array = np.stack([pixel_array for _, pixel_array in slices], axis=0).astype(np.float32)

    z_spacing = 1.0
    if len(z_positions) > 1:
        unique_positions = sorted({float(value) for value in z_positions})
        diffs = np.diff(unique_positions)
        if diffs.size > 0:
            z_spacing = float(np.median(np.abs(diffs)))
    elif fallback_thickness is not None:
        z_spacing = fallback_thickness

    if pixel_spacing is None:
        spacing = None
    else:
        spacing = (float(z_spacing), float(pixel_spacing[0]), float(pixel_spacing[1]))

    return VolumeData(array=_ensure_3d(array, path), spacing=spacing)


def _read_png_file(file: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required to load PNG stacks.") from exc

    with Image.open(file) as image:
        array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.float32)


def _load_png_stack(path: Path) -> VolumeData:
    image_files = sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in PNG_LIKE_SUFFIXES)
    if not image_files:
        raise ValueError(f"No PNG/JPG/TIFF/BMP slices found in directory: {path}")

    slices = [_read_png_file(file) for file in image_files]
    volume = np.stack(slices, axis=0).astype(np.float32)
    return VolumeData(array=_ensure_3d(volume, path), spacing=None)


def load_volume_data(path: Path) -> VolumeData:
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


def load_volume(path: Path) -> np.ndarray:
    return load_volume_data(path).array


def _spacing_matches(
    first: tuple[float, float, float] | None,
    second: tuple[float, float, float] | None,
    tolerance: float = 1e-3,
) -> bool:
    if first is None or second is None:
        return True
    return all(abs(a - b) <= tolerance for a, b in zip(first, second))


def load_case_with_metadata(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float] | None]:
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
            spacing = None
        else:
            image_data = load_volume_data(image_path)
            mask_data = load_volume_data(mask_path)
            image = image_data.array
            mask = mask_data.array
            spacing = image_data.spacing or mask_data.spacing
            if not _spacing_matches(image_data.spacing, mask_data.spacing):
                raise ValueError(f"Image/mask spacing mismatch for case {image_path.name}: {image_data.spacing} vs {mask_data.spacing}")
    else:
        image_data = load_volume_data(image_path)
        mask_data = load_volume_data(mask_path)
        image = image_data.array
        mask = mask_data.array
        spacing = image_data.spacing or mask_data.spacing
        if not _spacing_matches(image_data.spacing, mask_data.spacing):
            raise ValueError(f"Image/mask spacing mismatch for case {image_path.name}: {image_data.spacing} vs {mask_data.spacing}")

    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask shape mismatch for case {image_path.name}: {image.shape} vs {mask.shape}"
        )

    mask = (mask > 0).astype(np.float32)
    return image.astype(np.float32), mask, spacing


def load_case(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image, mask, _ = load_case_with_metadata(image_path, mask_path)
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
