import pandas as pd

df = pd.read_csv("data/metadata/dicom_metadata.csv")

print(df.shape)
print(df.head())
print(df.columns)
print(df["Manufacturer"].value_counts(dropna=False))
print(df["MagneticFieldStrength"].value_counts(dropna=False))
print(df[["EchoTime", "RepetitionTime"]].isna().mean())
print(df["SeriesDescription"].value_counts(dropna=False).head(50))
print(df["ProtocolName"].value_counts(dropna=False).head(50))