import argparse
from pathlib import Path

import pandas as pd
from idc_index import IDCClient


COLLECTION_ID = "duke_breast_cancer_mri"


def build_patient_download_plan(client, target_gb, seed):
    query = f"""
    SELECT
        collection_id,
        PatientID,
        StudyInstanceUID,
        SeriesInstanceUID,
        SeriesDescription,
        Modality,
        Manufacturer,
        series_size_MB
    FROM index
    WHERE collection_id = '{COLLECTION_ID}'
      AND Modality = 'MR'
      AND series_size_MB IS NOT NULL
    """

    series_df = client.sql_query(query)

    print(f"Total MR series found: {len(series_df)}")
    print(f"Total patients found: {series_df['PatientID'].nunique()}")
    print(f"Estimated total size: {series_df['series_size_MB'].sum() / 1024:.2f} GB")

    patient_df = (
        series_df.groupby("PatientID")
        .agg(
            total_size_MB=("series_size_MB", "sum"),
            num_series=("SeriesInstanceUID", "nunique"),
            manufacturers=("Manufacturer", lambda x: ", ".join(sorted(set(x.dropna().astype(str))))),
            example_series=("SeriesDescription", lambda x: " | ".join(list(x.dropna().astype(str).unique())[:5])),
        )
        .reset_index()
    )

    # Randomized but reproducible selection so we do not only get the first few patients.
    patient_df = patient_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    target_mb = target_gb * 1024
    selected_rows = []
    running_mb = 0.0

    for _, row in patient_df.iterrows():
        if running_mb + row["total_size_MB"] > target_mb:
            continue

        selected_rows.append(row)
        running_mb += row["total_size_MB"]

        if running_mb >= target_mb * 0.95:
            break

    selected_patients = pd.DataFrame(selected_rows)

    selected_series = series_df[series_df["PatientID"].isin(selected_patients["PatientID"])]

    return selected_patients, selected_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gb", type=float, default=50.0)
    parser.add_argument("--download_dir", type=str, default=r"D:\mri_data\duke_raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    client = IDCClient.client()

    selected_patients, selected_series = build_patient_download_plan(
        client=client,
        target_gb=args.target_gb,
        seed=args.seed,
    )

    patient_ids = selected_patients["PatientID"].tolist()

    print("\nSelected download plan")
    print("----------------------")
    print(f"Patients selected: {len(patient_ids)}")
    print(f"Series selected: {selected_series['SeriesInstanceUID'].nunique()}")
    print(f"Estimated size: {selected_series['series_size_MB'].sum() / 1024:.2f} GB")
    print(f"Download directory: {download_dir}")

    print("\nManufacturer distribution in selected series:")
    print(selected_series["Manufacturer"].value_counts(dropna=False))

    print("\nTop selected series descriptions:")
    print(selected_series["SeriesDescription"].value_counts(dropna=False).head(30))

    selected_patients.to_csv(reports_dir / "selected_patients_download_plan.csv", index=False)
    selected_series.to_csv(reports_dir / "selected_series_download_plan.csv", index=False)

    print("\nSaved:")
    print(reports_dir / "selected_patients_download_plan.csv")
    print(reports_dir / "selected_series_download_plan.csv")

    if args.dry_run:
        print("\nDry run only. No files will be downloaded.")
        client.download_from_selection(
            downloadDir=str(download_dir),
            collection_id=COLLECTION_ID,
            patientId=patient_ids,
            dry_run=True,
            quiet=False,
            show_progress_bar=True,
        )
        return

    print("\nStarting download...")
    client.download_from_selection(
        downloadDir=str(download_dir),
        collection_id=COLLECTION_ID,
        patientId=patient_ids,
        dry_run=False,
        quiet=False,
        show_progress_bar=True,
        use_s5cmd_sync=True,
    )

    print("\nDownload complete.")


if __name__ == "__main__":
    main()