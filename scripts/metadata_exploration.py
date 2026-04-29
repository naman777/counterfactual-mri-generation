from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METADATA_PATH = Path("data/metadata/dicom_metadata.csv")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_plot(series, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_hist_plot(series, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(10, 5))
    series.dropna().hist(bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = pd.read_csv(METADATA_PATH)

    numeric_cols = [
        "MagneticFieldStrength",
        "EchoTime",
        "RepetitionTime",
        "FlipAngle",
        "SliceThickness",
        "Rows",
        "Columns",
        "InstanceNumber",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_slices = len(df)
    num_patients = df["PatientID"].nunique()
    num_studies = df["StudyInstanceUID"].nunique()
    num_series = df["SeriesInstanceUID"].nunique()

    manufacturer_counts = df["Manufacturer"].value_counts(dropna=False)
    field_strength_counts = df["MagneticFieldStrength"].value_counts(dropna=False).sort_index()
    series_counts = df["SeriesDescription"].value_counts(dropna=False)

    missing_report = df[
        [
            "Manufacturer",
            "MagneticFieldStrength",
            "EchoTime",
            "RepetitionTime",
            "FlipAngle",
            "SliceThickness",
            "SeriesDescription",
            "ProtocolName",
        ]
    ].isna().mean() * 100

    te_summary = df["EchoTime"].describe()
    tr_summary = df["RepetitionTime"].describe()

    te_unique = df["EchoTime"].nunique()
    tr_unique = df["RepetitionTime"].nunique()

    by_manufacturer = df.groupby("Manufacturer")[["EchoTime", "RepetitionTime"]].agg(
        ["count", "mean", "std", "min", "max"]
    )

    by_field_strength = df.groupby("MagneticFieldStrength")[["EchoTime", "RepetitionTime"]].agg(
        ["count", "mean", "std", "min", "max"]
    )

    save_bar_plot(
        manufacturer_counts,
        "Manufacturer Distribution",
        "Manufacturer",
        "Number of slices",
        FIGURE_DIR / "manufacturer_distribution.png",
    )

    save_bar_plot(
        field_strength_counts,
        "Field Strength Distribution",
        "Field strength",
        "Number of slices",
        FIGURE_DIR / "field_strength_distribution.png",
    )

    save_hist_plot(
        df["EchoTime"],
        "Echo Time Distribution",
        "Echo Time",
        "Number of slices",
        FIGURE_DIR / "echo_time_distribution.png",
    )

    save_hist_plot(
        df["RepetitionTime"],
        "Repetition Time Distribution",
        "Repetition Time",
        "Number of slices",
        FIGURE_DIR / "repetition_time_distribution.png",
    )

    report_path = REPORT_DIR / "metadata_summary.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Metadata Exploration Summary\n\n")

        f.write("## Dataset scale\n")
        f.write(f"- Number of slices: {num_slices}\n")
        f.write(f"- Number of patients: {num_patients}\n")
        f.write(f"- Number of studies: {num_studies}\n")
        f.write(f"- Number of series: {num_series}\n\n")

        f.write("## Manufacturer distribution\n")
        f.write(manufacturer_counts.to_markdown())
        f.write("\n\n")

        f.write("## Field strength distribution\n")
        f.write(field_strength_counts.to_markdown())
        f.write("\n\n")

        f.write("## Missing value percentage\n")
        f.write(missing_report.to_markdown())
        f.write("\n\n")

        f.write("## Echo Time summary\n")
        f.write(te_summary.to_markdown())
        f.write(f"\n\nUnique EchoTime values: {te_unique}\n\n")

        f.write("## Repetition Time summary\n")
        f.write(tr_summary.to_markdown())
        f.write(f"\n\nUnique RepetitionTime values: {tr_unique}\n\n")

        f.write("## TE/TR grouped by manufacturer\n")
        f.write(by_manufacturer.to_markdown())
        f.write("\n\n")

        f.write("## TE/TR grouped by field strength\n")
        f.write(by_field_strength.to_markdown())
        f.write("\n\n")

        f.write("## Top series descriptions\n")
        f.write(series_counts.head(30).to_markdown())
        f.write("\n\n")

        f.write("## Decision\n")
        f.write("- Selected continuous parameter: TODO\n")
        f.write("- Reason: TODO\n")
        f.write("- Selected series for preprocessing: TODO\n")

    print(f"Saved report to {report_path}")
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()