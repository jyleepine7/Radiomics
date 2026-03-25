from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nsclc_unet.config import PipelineConfig
from nsclc_unet.spreadsheet import read_xlsx_sheet


CLINICAL_EXCLUDE_SUBSTRINGS = (
    "durable response",
    "pneumonitis",
    "ctcae",
    "date of pneumonitis",
    "progression",
    "last follow-up",
    "date of death or last f/u",
    "death during index tx",
    "death",
    "tx dc date",
    "exclusion criteria",
)


def _clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
            "private": np.nan,
        }
    )


def _coerce_series(series: pd.Series) -> pd.Series:
    cleaned = _clean_string_series(series)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    convertible_ratio = float(numeric.notna().mean()) if len(numeric) else 0.0
    if convertible_ratio >= 0.8:
        return numeric
    return cleaned


def _months_to_days(months: int) -> int:
    return {12: 365, 36: 1095}.get(months, int(round(months * 30.4375)))


def load_clinical_sheet(xlsx_path: Path) -> pd.DataFrame:
    clinical = read_xlsx_sheet(xlsx_path, "Clinical").copy()
    clinical = clinical.rename(columns={"study ID": "patient_id"})
    clinical["patient_id"] = clinical["patient_id"].astype(str).str.strip()
    clinical = clinical[clinical["patient_id"].notna() & (clinical["patient_id"] != "")]
    clinical = clinical.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    return clinical


def load_radiomics_sheet(xlsx_path: Path) -> pd.DataFrame:
    radiomics = read_xlsx_sheet(xlsx_path, "Radiomics").copy()
    first_column = radiomics.columns[0]
    radiomics = radiomics.rename(columns={first_column: "patient_id"})
    radiomics["patient_id"] = radiomics["patient_id"].astype(str).str.strip()
    radiomics = radiomics[radiomics["patient_id"].notna() & (radiomics["patient_id"] != "")]
    return radiomics.reset_index(drop=True)


def build_endpoint_table(clinical: pd.DataFrame, windows_months: tuple[int, ...]) -> pd.DataFrame:
    working = clinical.copy()
    progress_days = pd.to_numeric(_clean_string_series(working["Date of progression (days)"]), errors="coerce")
    os_days = pd.to_numeric(_clean_string_series(working["Date of death or last f/u"]), errors="coerce")
    death_event = pd.to_numeric(_clean_string_series(working["Death"]), errors="coerce").fillna(0).astype(int)
    progression_event = pd.to_numeric(_clean_string_series(working["progression"]), errors="coerce").fillna(0).astype(int)

    pfs_event = ((progression_event == 1) | (death_event == 1)).astype(int)
    pfs_days = progress_days.copy()
    missing_progression_time = pfs_days.isna()
    pfs_days.loc[missing_progression_time] = os_days.loc[missing_progression_time]
    both_event_times = progression_event.eq(1) & death_event.eq(1) & os_days.notna() & progress_days.notna()
    pfs_days.loc[both_event_times] = np.minimum(progress_days.loc[both_event_times], os_days.loc[both_event_times])

    endpoints = pd.DataFrame(
        {
            "patient_id": working["patient_id"],
            "time_os_days": os_days,
            "event_os": death_event,
            "time_pfs_days": pfs_days,
            "event_pfs": pfs_event,
        }
    )

    for months in windows_months:
        horizon_days = _months_to_days(months)

        os_label = np.where(
            endpoints["event_os"].eq(1) & endpoints["time_os_days"].le(horizon_days),
            1.0,
            np.where(endpoints["time_os_days"].ge(horizon_days), 0.0, np.nan),
        )
        pfs_label = np.where(
            endpoints["event_pfs"].eq(1) & endpoints["time_pfs_days"].le(horizon_days),
            1.0,
            np.where(endpoints["time_pfs_days"].ge(horizon_days), 0.0, np.nan),
        )

        endpoints[f"label_os_{months}m"] = os_label
        endpoints[f"eligible_os_{months}m"] = (~pd.isna(os_label)).astype(int)
        endpoints[f"label_pfs_{months}m"] = pfs_label
        endpoints[f"eligible_pfs_{months}m"] = (~pd.isna(pfs_label)).astype(int)

    return endpoints


def build_clinical_feature_table(clinical: pd.DataFrame) -> pd.DataFrame:
    feature_columns: list[str] = []
    for column in clinical.columns:
        lower = column.lower()
        if column == "patient_id":
            continue
        if any(token in lower for token in CLINICAL_EXCLUDE_SUBSTRINGS):
            continue
        series = _clean_string_series(clinical[column])
        if series.dropna().empty:
            continue
        if series.dropna().astype(str).str.len().max() > 80:
            continue
        if series.nunique(dropna=True) > 40:
            continue
        feature_columns.append(column)

    output = pd.DataFrame({"patient_id": clinical["patient_id"]})
    for column in feature_columns:
        output[f"clinical__{column}"] = _coerce_series(clinical[column])
    return output


def build_radiomics_feature_table(radiomics: pd.DataFrame, aggregations: tuple[str, ...]) -> pd.DataFrame:
    candidate_columns = [
        column
        for column in radiomics.columns
        if column.startswith(
            (
                "MORPHOLOGICAL_",
                "INTENSITY-BASED_",
                "INTENSITY-HISTOGRAM_",
                "LOCAL_INTENSITY_BASED_",
                "LOCAL_INTENSITY_HISTOGRAM_",
                "GLCM_",
                "GLRLM_",
                "NGTDM_",
                "GLSZM_",
            )
        )
    ]

    numeric_columns: dict[str, pd.Series] = {"patient_id": radiomics["patient_id"]}
    for column in candidate_columns:
        parsed = pd.to_numeric(_clean_string_series(radiomics[column]), errors="coerce")
        if parsed.notna().mean() < 0.8:
            continue
        numeric_columns[column] = parsed

    numeric_frame = pd.DataFrame(numeric_columns)

    grouped = numeric_frame.groupby("patient_id", as_index=False)
    pieces = [grouped["patient_id"].first()]

    if "mean" in aggregations:
        mean_df = grouped.mean(numeric_only=True).add_prefix("lifex_mean__")
        mean_df = mean_df.rename(columns={"lifex_mean__patient_id": "patient_id"})
        pieces.append(mean_df)
    if "max" in aggregations:
        max_df = grouped.max(numeric_only=True).add_prefix("lifex_max__")
        max_df = max_df.rename(columns={"lifex_max__patient_id": "patient_id"})
        pieces.append(max_df)
    if "std" in aggregations:
        std_df = grouped.std(numeric_only=True).fillna(0.0).add_prefix("lifex_std__")
        std_df = std_df.rename(columns={"lifex_std__patient_id": "patient_id"})
        pieces.append(std_df)

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="patient_id", how="left")
    return merged


def prepare_tabular_dataset(
    config: PipelineConfig,
    xlsx_path: Path | None = None,
    deep_features_path: Path | None = None,
) -> Path:
    workbook_path = xlsx_path or config.tabular.xlsx_path
    if workbook_path is None:
        raise ValueError("No XLSX path was provided. Set tabular.xlsx_path or pass --xlsx-path.")

    clinical = load_clinical_sheet(workbook_path)
    radiomics = load_radiomics_sheet(workbook_path)
    endpoints = build_endpoint_table(clinical, config.tabular.endpoint_windows_months)
    clinical_features = build_clinical_feature_table(clinical)
    radiomics_features = build_radiomics_feature_table(radiomics, config.tabular.radiomics_aggregations)

    prepared = endpoints.merge(clinical_features, on="patient_id", how="left")
    prepared = prepared.merge(radiomics_features, on="patient_id", how="left")

    resolved_deep_path = deep_features_path or config.feature_output_path
    if resolved_deep_path.exists():
        deep_features = pd.read_csv(resolved_deep_path)
        deep_features["patient_id"] = deep_features["patient_id"].astype(str).str.strip()
        prepared = prepared.merge(deep_features, on="patient_id", how="left")

    config.prepared_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(config.prepared_dataset_path, index=False)
    print(f"Saved prepared dataset to {config.prepared_dataset_path}")
    return config.prepared_dataset_path
