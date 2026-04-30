
"""python src/data/preprocess.py \
  --metadata_csv data/metadata/ax_t1_metadata.csv \
  --output_dir data/processed/ax_t1_128 \
  --image_size 128 \
  --save_format npy \
  --min_foreground 0.01 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """
    Find a column in df using case-insensitive matching.

    This makes the script work with metadata files that use either:
    PatientID / patient_id
    Manufacturer / manufacturer
    MagneticFieldStrength / magnetic_field_strength
    EchoTime / echo_time
    slice_path / SlicePath / path
    """
    lower_to_original = {c.lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]

    if required:
        raise ValueError(
            f"Could not find required column. Tried: {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def safe_float(value: Any) -> float:
    """Convert a value to float. Return NaN if conversion fails."""
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def normalize_manufacturer(value: Any) -> str:
    """Normalize manufacturer names into stable class names."""
    text = str(value).strip().upper()

    if "GE" in text:
        return "GE MEDICAL SYSTEMS"
    if "SIEMENS" in text:
        return "SIEMENS"

    return text


def normalize_field_strength(value: Any) -> str:
    """Normalize field strength values into stable class names."""
    val = safe_float(value)

    if math.isnan(val):
        return "UNKNOWN"

    # Common values are 1.5 and 3.0.
    if abs(val - 1.5) < 0.15:
        return "1.5T"
    if abs(val - 3.0) < 0.15:
        return "3.0T"

    return f"{val:g}T"


def apply_rescale_if_present(ds: pydicom.Dataset, image: np.ndarray) -> np.ndarray:
    """
    Apply DICOM RescaleSlope and RescaleIntercept if present.

    Some DICOM images store pixel values in encoded form.
    Applying these fields gives the displayed intensity scale.
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def percentile_clip_and_normalize(
    image: np.ndarray,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> np.ndarray:
    """
    Clip image intensities by percentiles and normalize to [0, 1].
    """
    image = image.astype(np.float32)

    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        raise ValueError("Image contains no finite pixels.")

    valid_pixels = image[finite_mask]
    low = np.percentile(valid_pixels, clip_low)
    high = np.percentile(valid_pixels, clip_high)

    if high <= low:
        raise ValueError(f"Invalid percentile range: low={low}, high={high}")

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    image = np.clip(image, 0.0, 1.0)

    return image.astype(np.float32)


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """
    Resize image to image_size x image_size.

    Uses cv2 if available. Falls back to skimage if cv2 is unavailable.
    """
    try:
        import cv2

        resized = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        from skimage.transform import resize

        resized = resize(
            image,
            (image_size, image_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )

    return resized.astype(np.float32)


def compute_foreground_fraction(image: np.ndarray, threshold: float = 0.05) -> float:
    """
    Estimate foreground fraction after normalization.

    Pixels above threshold are counted as foreground.
    Mostly-background edge slices will have a very small value.
    """
    return float((image > threshold).mean())


def read_dicom_image(path: Path) -> np.ndarray:
    """
    Load a DICOM file and return its normalized pixel array before resizing.
    """
    try:
        ds = pydicom.dcmread(str(path), force=True)
    except (InvalidDicomError, FileNotFoundError, PermissionError) as exc:
        raise ValueError(f"Could not read DICOM file: {path}. Error: {exc}") from exc

    if not hasattr(ds, "PixelData"):
        raise ValueError(f"No PixelData found in DICOM file: {path}")

    image = ds.pixel_array.astype(np.float32)
    image = apply_rescale_if_present(ds, image)

    return image.astype(np.float32)


def save_png(image: np.ndarray, output_path: Path) -> None:
    """
    Save normalized [0, 1] image as uint8 PNG.
    """
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise ImportError(
            "Saving PNG requires imageio. Install it with: pip install imageio"
        ) from exc

    arr = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    imageio.imwrite(output_path, arr)


def create_label_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create deterministic label maps for manufacturer and field strength.
    """
    manufacturers = sorted(df["manufacturer_clean"].dropna().unique().tolist())
    field_strengths = sorted(df["field_strength_clean"].dropna().unique().tolist())

    manufacturer_to_id = {name: idx for idx, name in enumerate(manufacturers)}
    field_strength_to_id = {name: idx for idx, name in enumerate(field_strengths)}

    return manufacturer_to_id, field_strength_to_id


def create_patient_splits(
    processed_df: pd.DataFrame,
    output_dir: Path,
    patient_col: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Create patient-level train/val/test split CSVs.

    This avoids data leakage because slices from the same patient cannot appear
    in multiple splits.
    """
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    patients = processed_df[patient_col].dropna().astype(str).unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_df = processed_df[processed_df[patient_col].astype(str).isin(train_patients)].copy()
    val_df = processed_df[processed_df[patient_col].astype(str).isin(val_patients)].copy()
    test_df = processed_df[processed_df[patient_col].astype(str).isin(test_patients)].copy()

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    summary = {
        "total_patients": n_total,
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),
    }

    with open(split_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_prepared_metadata(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Standardize important column names and remove rows with missing required values.
    """
    col_patient = find_column(raw_df, ["patient_id", "PatientID", "PatientId"])
    col_study = find_column(raw_df, ["study_uid", "StudyInstanceUID"], required=False)
    col_series = find_column(raw_df, ["series_uid", "SeriesInstanceUID"], required=False)
    col_sop = find_column(raw_df, ["sop_uid", "SOPInstanceUID"], required=False)
    col_manufacturer = find_column(raw_df, ["manufacturer", "Manufacturer"])
    col_field = find_column(raw_df, ["magnetic_field_strength", "MagneticFieldStrength", "field_strength"])
    col_echo = find_column(raw_df, ["echo_time", "EchoTime", "TE"])
    col_slice_path = find_column(raw_df, ["slice_path", "SlicePath", "path", "dicom_path", "filepath"])
    col_instance = find_column(raw_df, ["instance_number", "InstanceNumber"], required=False)
    col_series_desc = find_column(raw_df, ["series_description", "SeriesDescription"], required=False)
    col_protocol = find_column(raw_df, ["protocol_name", "ProtocolName"], required=False)

    cols = {
        "patient": col_patient,
        "study": col_study,
        "series": col_series,
        "sop": col_sop,
        "manufacturer": col_manufacturer,
        "field": col_field,
        "echo": col_echo,
        "slice_path": col_slice_path,
        "instance": col_instance,
        "series_desc": col_series_desc,
        "protocol": col_protocol,
    }

    df = raw_df.copy()
    df["patient_id"] = df[col_patient].astype(str)
    df["study_uid"] = df[col_study].astype(str) if col_study else ""
    df["series_uid"] = df[col_series].astype(str) if col_series else ""
    df["sop_uid"] = df[col_sop].astype(str) if col_sop else ""
    df["manufacturer_clean"] = df[col_manufacturer].apply(normalize_manufacturer)
    df["field_strength_clean"] = df[col_field].apply(normalize_field_strength)
    df["magnetic_field_strength"] = df[col_field].apply(safe_float)
    df["echo_time"] = df[col_echo].apply(safe_float)
    df["slice_path"] = df[col_slice_path].astype(str)
    df["instance_number"] = df[col_instance] if col_instance else np.nan
    df["series_description"] = df[col_series_desc].astype(str) if col_series_desc else ""
    df["protocol_name"] = df[col_protocol].astype(str) if col_protocol else ""

    before = len(df)
    df = df.replace({"": np.nan})
    df = df.dropna(
        subset=[
            "patient_id",
            "manufacturer_clean",
            "field_strength_clean",
            "magnetic_field_strength",
            "echo_time",
            "slice_path",
        ]
    ).copy()
    after = len(df)

    print(f"[INFO] Rows before required-field filtering: {before}")
    print(f"[INFO] Rows after required-field filtering:  {after}")
    print(f"[INFO] Dropped rows: {before - after}")

    return df, cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DICOM MRI slices.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Input metadata CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--image_size", type=int, default=128, help="Output image size.")
    parser.add_argument(
        "--save_format",
        type=str,
        default="npy",
        choices=["npy", "png"],
        help="Processed image format.",
    )
    parser.add_argument(
        "--min_foreground",
        type=float,
        default=0.01,
        help="Reject slices with foreground fraction below this threshold.",
    )
    parser.add_argument(
        "--foreground_threshold",
        type=float,
        default=0.05,
        help="Pixel threshold used to estimate foreground after normalization.",
    )
    parser.add_argument("--clip_low", type=float, default=1.0, help="Low percentile for clipping.")
    parser.add_argument("--clip_high", type=float, default=99.0, help="High percentile for clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient splits.")
    parser.add_argument(
        "--no_splits",
        action="store_true",
        help="Disable train/val/test split creation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )

    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    label_maps_dir = output_dir / "label_maps"

    images_dir.mkdir(parents=True, exist_ok=True)
    label_maps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading metadata: {metadata_csv}")
    raw_df = pd.read_csv(metadata_csv)

    if args.limit is not None:
        raw_df = raw_df.head(args.limit).copy()
        print(f"[INFO] Using first {args.limit} rows for debugging.")

    df, _ = build_prepared_metadata(raw_df)

    manufacturer_to_id, field_strength_to_id = create_label_maps(df)

    df["manufacturer_label"] = df["manufacturer_clean"].map(manufacturer_to_id).astype(int)
    df["field_strength_label"] = df["field_strength_clean"].map(field_strength_to_id).astype(int)

    echo_min = float(df["echo_time"].min())
    echo_max = float(df["echo_time"].max())
    echo_mean = float(df["echo_time"].mean())
    echo_std = float(df["echo_time"].std())

    if echo_max <= echo_min:
        raise ValueError("EchoTime has no variation. Cannot create normalized EchoTime.")

    df["echo_time_norm"] = (df["echo_time"] - echo_min) / (echo_max - echo_min)

    with open(label_maps_dir / "manufacturer_to_id.json", "w", encoding="utf-8") as f:
        json.dump(manufacturer_to_id, f, indent=2)

    with open(label_maps_dir / "field_strength_to_id.json", "w", encoding="utf-8") as f:
        json.dump(field_strength_to_id, f, indent=2)

    echo_stats = {
        "min": echo_min,
        "max": echo_max,
        "mean": echo_mean,
        "std": echo_std,
        "normalization": "min_max",
        "formula": "echo_time_norm = (echo_time - min) / (max - min)",
    }
    with open(label_maps_dir / "echo_time_stats.json", "w", encoding="utf-8") as f:
        json.dump(echo_stats, f, indent=2)

    processed_rows = []
    rejected_rows = []

    print("[INFO] Starting DICOM preprocessing...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        source_path = Path(str(row["slice_path"]))

        # Support both absolute and relative paths.
        if not source_path.exists():
            possible_path = metadata_csv.parent / source_path
            if possible_path.exists():
                source_path = possible_path

        try:
            image = read_dicom_image(source_path)
            image = percentile_clip_and_normalize(
                image,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
            )
            image = resize_image(image, args.image_size)

            foreground_fraction = compute_foreground_fraction(
                image,
                threshold=args.foreground_threshold,
            )

            if foreground_fraction < args.min_foreground:
                rejected_rows.append(
                    {
                        "row_index": idx,
                        "slice_path": str(row["slice_path"]),
                        "reason": "low_foreground",
                        "foreground_fraction": foreground_fraction,
                    }
                )
                continue

            patient_id_safe = str(row["patient_id"]).replace("/", "_").replace("\\", "_")
            sop_uid = str(row["sop_uid"])
            if sop_uid and sop_uid.lower() != "nan":
                stem = sop_uid.replace("/", "_").replace("\\", "_")
            else:
                stem = f"row_{idx:08d}"

            extension = ".npy" if args.save_format == "npy" else ".png"
            output_path = images_dir / f"{patient_id_safe}_{stem}{extension}"

            if args.save_format == "npy":
                np.save(output_path, image.astype(np.float32))
            else:
                save_png(image, output_path)

            processed_rows.append(
                {
                    "image_path": str(output_path),
                    "patient_id": row["patient_id"],
                    "study_uid": row["study_uid"],
                    "series_uid": row["series_uid"],
                    "sop_uid": row["sop_uid"],
                    "manufacturer": row["manufacturer_clean"],
                    "manufacturer_label": int(row["manufacturer_label"]),
                    "magnetic_field_strength": float(row["magnetic_field_strength"]),
                    "field_strength": row["field_strength_clean"],
                    "field_strength_label": int(row["field_strength_label"]),
                    "echo_time": float(row["echo_time"]),
                    "echo_time_norm": float(row["echo_time_norm"]),
                    "slice_path": str(row["slice_path"]),
                    "foreground_fraction": foreground_fraction,
                    "instance_number": row["instance_number"],
                    "series_description": row["series_description"],
                    "protocol_name": row["protocol_name"],
                }
            )

        except Exception as exc:
            rejected_rows.append(
                {
                    "row_index": idx,
                    "slice_path": str(row.get("slice_path", "")),
                    "reason": "processing_error",
                    "error": str(exc),
                }
            )

    processed_df = pd.DataFrame(processed_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    processed_metadata_path = output_dir / "processed_metadata.csv"
    rejected_metadata_path = output_dir / "rejected_slices.csv"

    processed_df.to_csv(processed_metadata_path, index=False)
    rejected_df.to_csv(rejected_metadata_path, index=False)

    print(f"[INFO] Saved processed metadata: {processed_metadata_path}")
    print(f"[INFO] Saved rejected-slice log:  {rejected_metadata_path}")
    print(f"[INFO] Processed slices: {len(processed_df)}")
    print(f"[INFO] Rejected slices:  {len(rejected_df)}")

    if len(processed_df) == 0:
        raise RuntimeError(
            "No slices were processed successfully. Check slice_path values and DICOM files."
        )

    summary = {
        "metadata_csv": str(metadata_csv),
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "save_format": args.save_format,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "foreground_threshold": args.foreground_threshold,
        "min_foreground": args.min_foreground,
        "input_rows_after_filtering": int(len(df)),
        "processed_slices": int(len(processed_df)),
        "rejected_slices": int(len(rejected_df)),
        "num_patients": int(processed_df["patient_id"].nunique()),
        "manufacturer_counts": processed_df["manufacturer"].value_counts().to_dict(),
        "field_strength_counts": processed_df["field_strength"].value_counts().to_dict(),
        "echo_time_min": float(processed_df["echo_time"].min()),
        "echo_time_max": float(processed_df["echo_time"].max()),
        "echo_time_mean": float(processed_df["echo_time"].mean()),
    }

    if not args.no_splits:
        split_summary = create_patient_splits(
            processed_df=processed_df,
            output_dir=output_dir,
            patient_col="patient_id",
            train_ratio=0.70,
            val_ratio=0.15,
            seed=args.seed,
        )
        summary["splits"] = split_summary

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Preprocessing complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
