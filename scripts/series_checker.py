import pandas as pd

df = pd.read_csv("data/metadata/dicom_metadata.csv")

for series_name in df["SeriesDescription"].value_counts().index:
    temp = df[df["SeriesDescription"] == series_name]

    print("\n" + "=" * 80)
    print("Series:", series_name)
    print("Slices:", len(temp))
    print("Patients:", temp["PatientID"].nunique())
    print("Manufacturers:", temp["Manufacturer"].value_counts().to_dict())
    print("Field strengths:", temp["MagneticFieldStrength"].value_counts().to_dict())
    print("EchoTime unique:", sorted(temp["EchoTime"].unique()))
    print("RepetitionTime unique:", sorted(temp["RepetitionTime"].unique()))