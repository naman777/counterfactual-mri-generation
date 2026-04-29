import pandas as pd
from pathlib import Path

input_csv = Path("data/metadata/dicom_metadata.csv")
output_csv = Path("data/metadata/clean_metadata.csv")

df = pd.read_csv(input_csv)

# Keep only actual scanner manufacturers for MVP
df = df[df["Manufacturer"].isin(["GE MEDICAL SYSTEMS", "SIEMENS"])]

# Remove rows with missing core acquisition metadata
required_cols = [
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "SliceThickness",
    "SeriesDescription",
    "slice_path",
]

df = df.dropna(subset=required_cols)

# Remove segmentation/label series
mask_seg = df["SeriesDescription"].str.contains(
    "segmentation|AIMI|radiologist",
    case=False,
    na=False,
)
df = df[~mask_seg]

output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print("Rows:", len(df))
print("Patients:", df["PatientID"].nunique())
print("Manufacturers:")
print(df["Manufacturer"].value_counts())
print("Field strengths:")
print(df["MagneticFieldStrength"].value_counts())