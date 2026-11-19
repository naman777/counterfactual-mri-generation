"""Look for resolvable axial T1 references in the existing annotation inventory."""

import json

import pandas as pd
import pydicom

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write

BASE = ROOT / "reports/v20_anatomy_preservation"


def main():
    data = ROOT / "data/metadata/ax_t1_metadata.csv"
    manifest = ROOT / "reports/v7_geometry/native_dce_v1_20261116_01/annotation_manifest.csv"
    mapping = ROOT / "reports/v4_downstream/mask_feasibility_20261016_01/segmentation_filepath_mapping.csv"
    ax_t1 = pd.read_csv(data, dtype=str)
    series_uids = set(ax_t1.SeriesInstanceUID)
    sop_uids = set(ax_t1.SOPInstanceUID)
    native = pd.read_csv(manifest, dtype=str)
    mapped = pd.read_csv(mapping, dtype=str)
    mapped_ax_t1 = mapped["Full Descriptive Path"].str.contains(r"/\d+(?:\.\d+)?-ax t1-", case=False, regex=True)
    paths = sorted(ROOT.glob("data/raw/**/SEG_*/*.dcm"))
    if not paths:
        raise ValueError("The local SEG inventory is unexpectedly empty")
    rows = []
    for path in paths:
        header = pydicom.dcmread(path, stop_before_pixels=True)
        if header.Modality != "SEG":
            raise ValueError("A file in the SEG inventory has a different modality")
        refs = {str(elem.value) for elem in header.iterall() if elem.keyword == "ReferencedSOPInstanceUID"}
        series = {str(elem.value) for elem in header.iterall() if elem.keyword == "SeriesInstanceUID"}
        series.discard(str(header.SeriesInstanceUID))
        rows.append(dict(path=path.relative_to(ROOT).as_posix(), patient_id=str(header.PatientID),
                         sha256=digest(path), referenced_sops=len(refs), referenced_series=len(series),
                         ax_t1_sop_matches=len(refs & sop_uids), ax_t1_series_matches=len(series & series_uids)))

    records = pd.DataFrame(rows)
    summary = dict(
        local_seg_files=len(rows), files_with_sop_references=int(records.referenced_sops.gt(0).sum()),
        files_with_ax_t1_sop_reference=int(records.ax_t1_sop_matches.gt(0).sum()),
        files_with_ax_t1_series_reference=int(records.ax_t1_series_matches.gt(0).sum()),
        native_annotation_frames=len(native), native_frame_ax_t1_sop_matches=int(native.source_sop_uid.isin(sop_uids).sum()),
        mapping_rows=len(mapped), mapping_exact_ax_t1_rows=int(mapped_ax_t1.sum()),
        annotated_generated_images=0, reviewed_generated_images=0,
    )
    out = BASE / "annotation_feasibility_20261118_01"
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out / "local_seg_inventory.csv", records)
    json_write(out / "summary.json", summary)
    json_write(out / "inputs.json", {p.relative_to(ROOT).as_posix(): digest(p) for p in (data, manifest, mapping)})
    (out / "README.md").write_text(
        "Producer: `python -m scripts.v20_annotation_audit`. `local_seg_inventory.csv` records "
        "existing local DICOM SEG header references and file hashes. `summary.json` records matches "
        "against the active axial T1 SOP and series inventory and the cross-check against the local native "
        "DCE manifest and the downloaded TCIA mapping file. `inputs.json` binds those tables. "
        "A UID match is only an eligibility signal and still needs frame-geometry and label verification. "
        "This audit used local metadata and read no source or annotation pixels.\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
