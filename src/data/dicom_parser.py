import argparse
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm


DICOM_FIELDS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Manufacturer",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "SliceThickness",
    "PixelSpacing",
    "Rows",
    "Columns",
    "InstanceNumber",
    "ImagePositionPatient",
    "SeriesDescription",
    "ProtocolName",
]


def safe_get(ds, field):
    value = getattr(ds, field, None)

    if value is None:
        return None

    # Convert DICOM multi-values to normal strings
    if isinstance(value, (list, tuple)):
        return "\\".join(str(v) for v in value)

    return str(value)


def find_dicom_files(input_dir):
    input_path = Path(input_dir)

    candidates = list(input_path.rglob("*.dcm"))

    # Some DICOM files may not have .dcm extension
    if len(candidates) == 0:
        candidates = [p for p in input_path.rglob("*") if p.is_file()]

    return candidates


def extract_metadata(input_dir):
    rows = []
    dicom_files = find_dicom_files(input_dir)

    print(f"Found {len(dicom_files)} candidate files")

    for path in tqdm(dicom_files):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True
            )

            row = {field: safe_get(ds, field) for field in DICOM_FIELDS}
            row["slice_path"] = str(path)
            rows.append(row)

        except Exception as e:
            rows.append({
                "slice_path": str(path),
                "error": str(e)
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = extract_metadata(args.input_dir)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")
    print(f"Rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()