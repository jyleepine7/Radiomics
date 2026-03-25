from __future__ import annotations

import csv
from pathlib import Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_feature_tables(
    primary_csv: Path,
    secondary_csv: Path,
    output_csv: Path,
    key: str = "patient_id",
) -> Path:
    primary_rows = read_csv_rows(primary_csv)
    secondary_rows = read_csv_rows(secondary_csv)
    secondary_lookup = {row[key]: row for row in secondary_rows if row.get(key)}

    merged_rows: list[dict[str, str]] = []
    for row in primary_rows:
        patient_id = row.get(key)
        if not patient_id:
            continue
        merged = dict(row)
        other = secondary_lookup.get(patient_id, {})
        for field, value in other.items():
            if field == key:
                continue
            merged[field] = value
        merged_rows.append(merged)

    if not merged_rows:
        raise ValueError("No rows were merged.")

    fieldnames: list[str] = []
    seen = set()
    for row in merged_rows:
        for field in row.keys():
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return output_csv
