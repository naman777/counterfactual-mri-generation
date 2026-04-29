from idc_index import IDCClient

client = IDCClient.client()

query = """
SELECT
    collection_id,
    PatientID,
    SeriesInstanceUID,
    SeriesDescription,
    Modality,
    series_size_MB
FROM index
WHERE collection_id = 'duke_breast_cancer_mri'
  AND Modality = 'MR'
LIMIT 20
"""

results = client.sql_query(query)

print(results[[
    "PatientID",
    "SeriesInstanceUID",
    "SeriesDescription",
    "series_size_MB"
]])

series_uids = results["SeriesInstanceUID"].tolist()

client.download_dicom_series(
    seriesInstanceUID=series_uids,
    downloadDir="data/raw"
)