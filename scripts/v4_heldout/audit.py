"""Local B2/B5 preflight; reads cached metadata and never downloads images.

Run from the code root: python -m scripts.v4_heldout.audit --run-id preflight_20261016_01
"""
import argparse
import hashlib
import json
from pathlib import Path

import idc_index_data
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in run_id):
        raise ValueError('Invalid run ID')
    out = ROOT / 'reports/v4_heldout_judge' / run_id
    out.mkdir(parents=True, exist_ok=False)
    splitpaths = [ROOT / f'data/processed/ax_t1_128/splits/{s}.csv' for s in ['train', 'val', 'test']]
    slices = pd.concat([pd.read_csv(p).assign(original_split=p.stem) for p in splitpaths], ignore_index=True)
    assert slices.groupby('patient_id').original_split.nunique().max() == 1
    patients = slices.drop_duplicates('patient_id').copy()
    used = set(patients.patient_id)
    assert len(used) == 103
    patients[['patient_id', 'original_split', 'manufacturer_label', 'field_strength_label']].to_csv(out / 'excluded_original_patients.csv', index=False)

    metadata_path = ROOT / 'data/metadata/clean_metadata.csv'
    metadata = pd.read_csv(metadata_path, usecols=['PatientID', 'SeriesDescription', 'Manufacturer', 'MagneticFieldStrength'], low_memory=False)
    extras = metadata[~metadata.PatientID.isin(used)]
    local_extra = extras.groupby('PatientID').agg(
        descriptions=('SeriesDescription', lambda x: ' | '.join(sorted(set(x.dropna().astype(str))))),
        manufacturers=('Manufacturer', lambda x: ' | '.join(sorted(set(x.dropna().astype(str))))),
        exact_ax_t1_slices=('SeriesDescription', lambda x: x.fillna('').str.strip().str.lower().eq('ax t1').sum()),
    ).reset_index()
    local_extra.to_csv(out / 'local_unused_patients.csv', index=False)

    clinical_path = ROOT / 'data/full/archive/Clinical_and_Other_Features.xlsx'
    clinical = pd.read_excel(clinical_path, sheet_name='Data', header=1).dropna(subset=['Patient ID'])
    assert clinical['Patient ID'].is_unique
    # The spreadsheet uses category codes, NOT literal Tesla values.
    clinical['planning_field_strength_label'] = pd.to_numeric(clinical['Field Strength (Tesla)'], errors='coerce').map({0: 0, 1: 0, 2: 1, 3: 1})
    clinical['planning_manufacturer_label'] = pd.to_numeric(clinical.Manufacturer, errors='coerce').map({0: 0, 2: 1})
    check = patients.merge(clinical[['Patient ID', 'planning_field_strength_label', 'planning_manufacturer_label']], left_on='patient_id', right_on='Patient ID', how='left', validate='one_to_one')
    check[['patient_id', 'manufacturer_label', 'planning_manufacturer_label', 'field_strength_label', 'planning_field_strength_label']].to_csv(out / 'clinical_encoding_check_original.csv', index=False)

    index_path = Path(idc_index_data.IDC_INDEX_PARQUET_FILEPATH)
    catalog = pd.read_parquet(index_path, filters=[('collection_id', '==', 'duke_breast_cancer_mri')])
    catalog['series_size_MB'] = pd.to_numeric(catalog.series_size_MB, errors='raise')
    candidate = catalog[catalog.Modality.eq('MR') & catalog.SeriesDescription.fillna('').str.strip().str.lower().eq('ax t1') & ~catalog.PatientID.isin(used)].copy()
    candidate['manufacturer_label'] = candidate.Manufacturer.map({'GE MEDICAL SYSTEMS': 0, 'SIEMENS': 1})
    candidate = candidate.merge(clinical[['Patient ID', 'planning_field_strength_label', 'planning_manufacturer_label']], left_on='PatientID', right_on='Patient ID', how='left', validate='many_to_one')
    candidate['eligible'] = candidate.manufacturer_label.notna() & candidate.planning_field_strength_label.notna() & candidate.manufacturer_label.eq(candidate.planning_manufacturer_label)
    candidate.to_csv(out / 'catalog_candidates.csv', index=False)
    eligible = candidate[candidate.eligible].copy()
    # A single series per patient, independent of clinical endpoints and size.
    eligible = eligible.sort_values(['PatientID', 'SeriesInstanceUID']).drop_duplicates('PatientID')
    eligible['selection_rank'] = eligible.PatientID.map(lambda p: hashlib.sha256(f'heldout-42:{p}'.encode()).hexdigest())
    selected = []
    for m in range(2):
        for f in range(2):
            group = eligible[eligible.manufacturer_label.eq(m) & eligible.planning_field_strength_label.eq(f)].sort_values('selection_rank')
            if len(group) < 16:
                raise ValueError(f'Insufficient cell mfg={m}, fs={f}: {len(group)}')
            cell = group.head(16).copy()
            cell['judge_split'] = ['train'] * 12 + ['val'] * 4
            selected.append(cell)
    plan = pd.concat(selected, ignore_index=True)
    assert plan.PatientID.is_unique and not (set(plan.PatientID) & used)
    assert plan.groupby('PatientID').judge_split.nunique().max() == 1
    assert len(plan) == 64 and int(plan.instanceCount.sum()) > 0
    plan.to_csv(out / 'download_plan.csv', index=False)
    summary = plan.groupby(['judge_split', 'manufacturer_label', 'planning_field_strength_label'], observed=True).agg(patients=('PatientID', 'nunique'), instances=('instanceCount', 'sum'), series_size_MB=('series_size_MB', 'sum')).reset_index()
    summary.to_csv(out / 'download_plan_counts.csv', index=False)

    labels_path = ROOT / 'data/full/archive/Patient class labels.csv'
    labels = pd.read_csv(labels_path)
    assert labels['Patient ID'].is_unique
    endpoints = ['ER', 'PR', 'HER2', 'Mol Subtype']
    labelled = patients.merge(labels[['Patient ID'] + endpoints], left_on='patient_id', right_on='Patient ID', how='left', validate='one_to_one')
    assert labelled[endpoints].notna().all().all()
    manifestcols = ['patient_id', 'original_split', 'manufacturer_label', 'field_strength_label'] + endpoints
    labelled[manifestcols].to_csv(out / 'downstream_patient_labels.csv', index=False)
    image_labels = slices.merge(labels[['Patient ID'] + endpoints], left_on='patient_id', right_on='Patient ID', how='left', validate='many_to_one')
    image_labels[['image_path', 'patient_id', 'original_split', 'manufacturer_label', 'field_strength_label'] + endpoints].to_csv(out / 'downstream_image_labels.csv', index=False)
    counts = []
    for endpoint in endpoints:
        rows = labelled.groupby(['original_split', 'manufacturer_label', endpoint], observed=True).size().reset_index(name='patients').rename(columns={endpoint: 'class'})
        rows['endpoint'] = endpoint
        counts.append(rows)
    pd.concat(counts, ignore_index=True).to_csv(out / 'downstream_label_counts.csv', index=False)

    seg = catalog[catalog.Modality.eq('SEG')].copy()
    seg['in_original_cohort'] = seg.PatientID.isin(used)
    seg['in_proposed_judge_cohort'] = seg.PatientID.isin(plan.PatientID)
    seg.to_csv(out / 'catalog_segmentation_inventory.csv', index=False)
    seg.groupby(['SeriesDescription', 'in_original_cohort'], observed=True).agg(patients=('PatientID', 'nunique'), series=('SeriesInstanceUID', 'nunique'), series_size_MB=('series_size_MB', 'sum')).reset_index().to_csv(out / 'segmentation_counts.csv', index=False)
    masks = [str(p.relative_to(ROOT)) for p in (ROOT / 'data').rglob('*') if p.is_file() and (p.name.lower().endswith(('.nii', '.nii.gz', '.nrrd', '.mha')))]
    (out / 'local_volumetric_masks.json').write_text(json.dumps(masks, indent=2), encoding='utf-8')
    inputs = splitpaths + [metadata_path, clinical_path, labels_path, index_path, Path(__file__)]
    manifest = {'index_package_version': idc_index_data.__version__, 'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in inputs]}
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    info = {
        'run_id': run_id, 'downloaded': False, 'cloud_jobs_launched': False,
        'original_patients_excluded': len(used), 'local_unused_patients': len(local_extra),
        'local_unused_exact_ax_t1_slices': int(local_extra.exact_ax_t1_slices.sum()),
        'catalog_exact_ax_t1_additional_patients': int(candidate.PatientID.nunique()),
        'eligible_additional_patients': int(eligible.PatientID.nunique()),
        'clinical_field_encoding_agreement_original': int(check.field_strength_label.eq(check.planning_field_strength_label).sum()),
        'selected_patients': len(plan), 'train_patients': int(plan.judge_split.eq('train').sum()),
        'validation_patients': int(plan.judge_split.eq('val').sum()),
        'instances_before_preprocessing': int(plan.instanceCount.sum()),
        'catalog_size_MB': float(plan.series_size_MB.sum()),
        'catalog_size_decimal_GB': float(plan.series_size_MB.sum() / 1000),
        'catalog_size_provenance': 'IDC cached v23 catalogue series_size_MB, an estimate; actual transfer may differ',
        'download_plan_sha256': sha(out / 'download_plan.csv'),
        'clinical_endpoint_selected': None, 'clinical_utility_established': False,
        'local_volumetric_mask_files_found': len(masks),
    }
    (out / 'summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(f'''# B2 and B5 local preflight: {run_id}

Created by `python -m scripts.v4_heldout.audit --run-id {run_id}`. This is a feasibility audit, not a judge-training or clinical-utility result. No DICOM images were downloaded and no cloud job was launched.

## B2 download proposal

The catalogue contains {info['catalog_exact_ax_t1_additional_patients']} additional exact `ax t1` patients after excluding all 103 original patients. The 20 unused local patients contain no exact `ax t1` slices (`local_unused_patients.csv`), so other sequences are not substituted.

`download_plan.csv` specifies 64 patients, 16 per manufacturer/field-strength cell: 48 train and 16 validation patients. Selection uses SHA-256 rank of `heldout-42:PatientID`; no receptor endpoint, test score or image-quality outcome determines selection. Within each cell the first 12 are training and last 4 validation. One series per patient is chosen by sorted series UID. `download_plan_counts.csv` has counts and size. Planned total: {info['instances_before_preprocessing']} instances, {info['catalog_size_MB']:.3f} catalogue MB ({info['catalog_size_decimal_GB']:.3f} decimal GB), plus working space for preprocessing. The cache is IDC package {idc_index_data.__version__}; availability and headers must be verified at download time. No full catalogue download is needed.

Field strength is a **planning label** from the local clinical spreadsheet: codes 0/1 map to 1.5T and 2/3 to 3T. These codes agree with DICOM-derived labels in {info['clinical_field_encoding_agreement_original']}/103 original patients (`clinical_encoding_check_original.csv`). Verify every new series' DICOM manufacturer, field strength, sequence, patient and series UID before inclusion. On any discrepancy, stop and document it; never silently relabel to force balance.

`excluded_original_patients.csv` freezes exclusion of train, validation AND test patients. `catalog_candidates.csv` preserves eligible and ineligible exact-sequence candidates. `input_manifest.json` records input hashes. `summary.json` is the machine-readable receipt. All files in this folder were produced by `audit.py`.

## Subsequent judge protocol

Reuse the existing preprocessing functions (rescale, per-slice percentile 1–99 clipping, [0,1], area resize 128, foreground threshold 0.01), original label maps, and original echo normalisation if retained. Write to a new run-specific processed folder. Preserve this planned patient split; do not call the default random-split preprocessing CLI. Train robust ResNet-18 with three seeds only on the new training patients and select epochs on new validation patients. Evaluate original 934 test slices only after freezing checkpoints; report both slice and patient-clustered accuracy CIs, class balance and per-direction scores. Regrade frozen old/new counterfactuals with these judges. The existing Modal judge script hardcodes `splits128.npz` and output names, so a versioned copy must accept explicit input/output namespaces before launch. Do not run it as-is. There is one B1 app active and no extra cloud budget is assumed.

## B5 feasibility

`downstream_patient_labels.csv` and `downstream_image_labels.csv` join actual ER, PR, HER2 and molecular-subtype labels to the existing frozen splits. `downstream_label_counts.csv` reports patient counts by split/vendor/class. Every original patient has these labels. Counts are descriptive feasibility data, not classifier performance. Do not treat slices as independent clinical subjects or select endpoints using test performance. Siemens training contains very few negative ER/PR patients, so this cohort is inadequate for a convincing broad clinical-utility claim without a power/sample-size assessment and stronger data. No clinical endpoint has been selected.

`local_volumetric_masks.json` inventories locally available NIfTI/NRRD/MHA files. The existing NIFTI annotation CSV describes bounding boxes, not voxel tissue masks. `catalog_segmentation_inventory.csv` and `segmentation_counts.csv` inventory catalogue DICOM SEG entries and cohort overlap. Descriptions include AI/corrected tumour annotations and generic `Segmentation`; descriptions alone do not establish breast/FGT anatomy, reference sequence or ground-truth quality. Do not substitute tumour boxes/AI pseudo-labels for breast/FGT masks. Inspect SegmentSequence and referenced series, then validate spatial registration before proposing a segmentation experiment. No mask download is authorised here.

Downstream augmentation must use B1's selected final checkpoint and frozen B3 settings, with real-only, ordinary augmentation, and counterfactual augmentation arms on the same patients/seeds. Keep clinical labels fixed, aggregate slice predictions per patient, use patient bootstrap CIs and paired patient comparisons, and report both source-vendor ID and target-vendor OOD results. A label being different from vendor does not prove independence from vendor; assess confounding using training subjects and report limitations. B5 remains pending.
''', encoding='utf-8')
    print(json.dumps(info, indent=2))
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
