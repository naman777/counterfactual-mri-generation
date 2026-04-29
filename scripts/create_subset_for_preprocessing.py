import pandas as pd
from pathlib import Path

df = pd.read_csv("data/metadata/clean_metadata.csv")

subset = df[df["SeriesDescription"] == "ax t1"].copy()

Path("data/metadata").mkdir(parents=True, exist_ok=True)
subset.to_csv("data/metadata/ax_t1_metadata.csv", index=False)

print("Rows:", len(subset))
print("Patients:", subset["PatientID"].nunique())
print("Manufacturers:")
print(subset["Manufacturer"].value_counts())
print("Field strengths:")
print(subset["MagneticFieldStrength"].value_counts())
print("EchoTime unique:", subset["EchoTime"].nunique())