from __future__ import annotations

import argparse
import csv
from pathlib import Path


SUPPORTED_FILE_SUFFIXES = (".nii.gz", ".nii", ".npy", ".npz")
SUPPORTED_STACK_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a manifest.csv from image and mask roots.")
    parser.add_argument("--images-root", required=True, help="Directory containing patient image volumes or patient slice folders.")
    parser.add_argument("--masks-root", required=True, help="Directory containing patient mask volumes or patient slice folders.")
    parser.add_argument("--output", required=True, help="Output manifest CSV path.")
    parser.add_argument(
        "--image-suffix",
        default="",
        help="Optional suffix to strip from image basenames when deriving patient_id, e.g. _ct.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="",
        help="Optional suffix to strip from mask basenames when deriving patient_id, e.g. _mask.",
    )
    return parser


def _basename_without_supported_suffix(path: Path) -> str:
    name = path.name
    for suffix in SUPPORTED_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _derive_patient_id(path: Path, suffix_to_strip: str) -> str:
    if path.is_dir():
        patient_id = path.name
    else:
        patient_id = _basename_without_supported_suffix(path)

    if suffix_to_strip and patient_id.endswith(suffix_to_strip):
        patient_id = patient_id[: -len(suffix_to_strip)]
    return patient_id


def _is_supported_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_FILE_SUFFIXES)


def _is_supported_stack_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(file.is_file() and file.suffix.lower() in SUPPORTED_STACK_SUFFIXES for file in path.iterdir())


def _collect_entries(root: Path, suffix_to_strip: str) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    entries: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            if not _is_supported_stack_dir(child):
                continue
            patient_id = _derive_patient_id(child, suffix_to_strip)
            entries[patient_id] = child.resolve()
            continue

        if not child.is_file() or not _is_supported_file(child):
            continue
        patient_id = _derive_patient_id(child, suffix_to_strip)
        entries[patient_id] = child.resolve()

    if not entries:
        raise ValueError(f"No supported image or mask entries found in {root}")
    return entries


def main() -> None:
    args = build_parser().parse_args()
    images_root = Path(args.images_root).resolve()
    masks_root = Path(args.masks_root).resolve()
    output_path = Path(args.output).resolve()

    image_entries = _collect_entries(images_root, args.image_suffix)
    mask_entries = _collect_entries(masks_root, args.mask_suffix)

    shared_patient_ids = sorted(set(image_entries).intersection(mask_entries))
    if not shared_patient_ids:
        raise ValueError("No matching patient IDs were found between images and masks.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "image_path", "mask_path"])
        writer.writeheader()
        for patient_id in shared_patient_ids:
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "image_path": str(image_entries[patient_id]),
                    "mask_path": str(mask_entries[patient_id]),
                }
            )

    print(f"Saved manifest with {len(shared_patient_ids)} patients to {output_path}")


if __name__ == "__main__":
    main()
