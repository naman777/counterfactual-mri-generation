#!/usr/bin/env python3
"""
Check patient-level train/val/test splits after MRI preprocessing.

Purpose:
- Confirm there is no patient leakage across train/val/test.
- Confirm all image_path files exist.
- Show manufacturer distribution in each split.
- Show field-strength distribution in each split.
- Show EchoTime range in each split.
- Save a JSON and Markdown report.

Example:
python scripts/check_splits.py \
  --processed_metadata data/processed/ax_t1_128/processed_metadata.csv \
  --split_dir data/processed/ax_t1_128/splits \
  --output_dir reports/preprocessing/ax_t1_128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def patient_set(df: pd.DataFrame) -> Set[str]:
    if "patient_id" not in df.columns:
        raise ValueError("Expected column not found: patient_id")
    return set(df["patient_id"].astype(str).unique().tolist())


def check_required_columns(df: pd.DataFrame, name: str) -> None:
    required = [
        "image_path",
        "patient_id",
        "manufacturer",
        "manufacturer_label",
        "field_strength",
        "field_strength_label",
        "echo_time",
        "echo_time_norm",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def distribution(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}


def echo_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "min": float(df["echo_time"].min()),
        "max": float(df["echo_time"].max()),
        "mean": float(df["echo_time"].mean()),
        "std": float(df["echo_time"].std()),
    }


def missing_files(df: pd.DataFrame) -> List[str]:
    missing = []
    for p in df["image_path"].astype(str).tolist():
        if not Path(p).exists():
            missing.append(p)
    return missing


def split_summary(df: pd.DataFrame, name: str) -> Dict:
    return {
        "split": name,
        "num_images": int(len(df)),
        "num_patients": int(df["patient_id"].astype(str).nunique()),
        "manufacturer_counts": distribution(df, "manufacturer"),
        "field_strength_counts": distribution(df, "field_strength"),
        "echo_time": echo_stats(df),
        "missing_image_files_count": len(missing_files(df)),
    }


def make_markdown_report(report: Dict) -> str:
    lines = []
    lines.append("# Split Sanity Check Report")
    lines.append("")
    lines.append("## Overall Status")
    lines.append("")
    lines.append(f"- Patient leakage check: **{report['patient_leakage_status']}**")
    lines.append(f"- Missing image files: **{report['total_missing_image_files']}**")
    lines.append(f"- Processed metadata rows: **{report['processed_metadata_rows']}**")
    lines.append("")

    lines.append("## Patient Counts")
    lines.append("")
    lines.append("| Split | Images | Patients |")
    lines.append("|---|---:|---:|")
    for s in ["train", "val", "test"]:
        info = report["splits"][s]
        lines.append(f"| {s} | {info['num_images']} | {info['num_patients']} |")
    lines.append("")

    lines.append("## Manufacturer Distribution")
    lines.append("")
    lines.append("| Split | Distribution |")
    lines.append("|---|---|")
    for s in ["train", "val", "test"]:
        lines.append(f"| {s} | `{report['splits'][s]['manufacturer_counts']}` |")
    lines.append("")

    lines.append("## Field Strength Distribution")
    lines.append("")
    lines.append("| Split | Distribution |")
    lines.append("|---|---|")
    for s in ["train", "val", "test"]:
        lines.append(f"| {s} | `{report['splits'][s]['field_strength_counts']}` |")
    lines.append("")

    lines.append("## EchoTime Statistics")
    lines.append("")
    lines.append("| Split | Min | Max | Mean | Std |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in ["train", "val", "test"]:
        e = report["splits"][s]["echo_time"]
        lines.append(
            f"| {s} | {e['min']:.4f} | {e['max']:.4f} | {e['mean']:.4f} | {e['std']:.4f} |"
        )
    lines.append("")

    lines.append("## Patient Overlap Details")
    lines.append("")
    lines.append(f"- Train-Val overlap: `{report['patient_overlaps']['train_val']}`")
    lines.append(f"- Train-Test overlap: `{report['patient_overlaps']['train_test']}`")
    lines.append(f"- Val-Test overlap: `{report['patient_overlaps']['val_test']}`")
    lines.append("")

    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("## Warnings")
        lines.append("")
        lines.append("- None")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MRI preprocessing splits.")
    parser.add_argument("--processed_metadata", required=True, type=str)
    parser.add_argument("--split_dir", required=True, type=str)
    parser.add_argument("--output_dir", default="reports/preprocessing/ax_t1_128", type=str)
    args = parser.parse_args()

    processed_metadata = Path(args.processed_metadata)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_df = load_csv(processed_metadata)
    train_df = load_csv(split_dir / "train.csv")
    val_df = load_csv(split_dir / "val.csv")
    test_df = load_csv(split_dir / "test.csv")

    for name, df in [
        ("processed_metadata", processed_df),
        ("train.csv", train_df),
        ("val.csv", val_df),
        ("test.csv", test_df),
    ]:
        check_required_columns(df, name)

    train_patients = patient_set(train_df)
    val_patients = patient_set(val_df)
    test_patients = patient_set(test_df)

    overlap_train_val = sorted(train_patients & val_patients)
    overlap_train_test = sorted(train_patients & test_patients)
    overlap_val_test = sorted(val_patients & test_patients)

    leakage_ok = (
        len(overlap_train_val) == 0
        and len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
    )

    all_missing_files = (
        missing_files(train_df)
        + missing_files(val_df)
        + missing_files(test_df)
    )

    warnings = []

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        manufacturer_classes = df["manufacturer"].dropna().astype(str).nunique()
        field_strength_classes = df["field_strength"].dropna().astype(str).nunique()

        if manufacturer_classes < 2:
            warnings.append(
                f"{split_name} split has fewer than 2 manufacturer classes. "
                "Predictor validation for manufacturer may be weak."
            )

        if field_strength_classes < 2:
            warnings.append(
                f"{split_name} split has fewer than 2 field-strength classes. "
                "Predictor validation for field strength may be weak."
            )

        if df["echo_time"].nunique() < 2:
            warnings.append(
                f"{split_name} split has fewer than 2 EchoTime values. "
                "EchoTime regression validation may be weak."
            )

    if not leakage_ok:
        warnings.append("Patient leakage detected across splits. Regenerate patient-level splits.")

    if all_missing_files:
        warnings.append(f"{len(all_missing_files)} image files are missing from disk.")

    report = {
        "processed_metadata_rows": int(len(processed_df)),
        "patient_leakage_status": "PASS" if leakage_ok else "FAIL",
        "patient_overlaps": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test,
        },
        "total_missing_image_files": int(len(all_missing_files)),
        "splits": {
            "train": split_summary(train_df, "train"),
            "val": split_summary(val_df, "val"),
            "test": split_summary(test_df, "test"),
        },
        "warnings": warnings,
    }

    json_path = output_dir / "split_sanity_report.json"
    md_path = output_dir / "split_sanity_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(make_markdown_report(report), encoding="utf-8")

    print("\n=== Split Sanity Check ===")
    print(f"Patient leakage: {report['patient_leakage_status']}")
    print(f"Missing image files: {report['total_missing_image_files']}")
    print("\nImage/patient counts:")
    for s in ["train", "val", "test"]:
        info = report["splits"][s]
        print(f"  {s}: {info['num_images']} images, {info['num_patients']} patients")

    print("\nManufacturer distribution:")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {report['splits'][s]['manufacturer_counts']}")

    print("\nField strength distribution:")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {report['splits'][s]['field_strength_counts']}")

    print("\nEchoTime stats:")
    for s in ["train", "val", "test"]:
        e = report["splits"][s]["echo_time"]
        print(
            f"  {s}: min={e['min']:.4f}, max={e['max']:.4f}, "
            f"mean={e['mean']:.4f}, std={e['std']:.4f}"
        )

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nWarnings: none")

    print(f"\nSaved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")


if __name__ == "__main__":
    main()
