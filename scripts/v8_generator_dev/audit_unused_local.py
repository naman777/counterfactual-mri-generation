"""Inventory unused local patients by metadata/header only; never expands a split."""
import argparse
import csv
import json
import re
from collections import defaultdict

import pandas as pd
import pydicom

from scripts.v7_geometry.orientation import orientation_plan
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def audit(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new audit ID required')
    used, inputs = set(), {}
    # Test membership IDs are read only to exclude those patients from new development.
    for split in ('train', 'val', 'test'):
        path = ROOT / 'data/processed/ax_t1_128/splits' / f'{split}.csv'
        used.update(pd.read_csv(path, usecols=['patient_id']).patient_id.astype(str))
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    root = ROOT / 'data/raw/duke_breast_cancer_mri'
    local = {path.name for path in root.glob('Breast_MRI_*') if path.is_dir()}
    unused = local-used
    groups, provenance = {}, defaultdict(set)
    for filename in ('ax_t1_metadata.csv', 'clean_metadata.csv', 'dicom_metadata.csv'):
        path = ROOT / 'data/metadata' / filename
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        with path.open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                patient = row['PatientID']
                if patient not in unused:
                    continue
                key = (patient, row['SeriesInstanceUID'])
                provenance[key].add(filename)
                group = groups.setdefault(key, dict(patient_id=patient, series_uid=key[1],
                    series_description=row['SeriesDescription'], manufacturer=row['Manufacturer'],
                    field_strength=row['MagneticFieldStrength'], echo_time=row['EchoTime'],
                    metadata_files=set(), sop_uids=set(), representative_path=row['slice_path']))
                group['sop_uids'].add(row['SOPInstanceUID'])
    rows = []
    for key, value in groups.items():
        value = dict(value)
        value.pop('metadata_files')
        value['metadata_sources'] = '|'.join(sorted(provenance[key]))
        value['slices_in_metadata'] = len(value.pop('sop_uids'))
        value['exact_current_ax_t1_description'] = value['series_description'].strip().lower() == 'ax t1'
        value['header_orientation_compatible'] = False
        value['header_status'] = 'not_checked_nonmatching_sequence'
        if value['exact_current_ax_t1_description']:
            try:
                path = project_path(value['representative_path'])
                header = pydicom.dcmread(path, stop_before_pixels=True)
                orientation_plan(list(header.ImageOrientationPatient), (int(header.Rows), int(header.Columns)))
                value['header_orientation_compatible'] = True
                value['header_status'] = 'one_header_compatible_requires_full_series_and_protocol_audit'
                inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            except (ValueError, AttributeError, FileNotFoundError) as error:
                value['header_status'] = str(error)
        rows.append(value)
    patients = []
    for patient in sorted(unused):
        matching = [row for row in rows if row['patient_id'] == patient and row['exact_current_ax_t1_description']]
        patients.append(dict(patient_id=patient, metadata_series=sum(row['patient_id']==patient for row in rows),
            exact_current_ax_t1_series=len(matching),
            compatible_sample_headers=sum(row['header_orientation_compatible'] for row in matching),
            added_to_development=False, evidence_status='candidate_only' if matching else 'no_matching_ax_t1_in_existing_metadata'))
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    csv_write(output / 'unused_patients.csv', pd.DataFrame(patients))
    csv_write(output / 'unused_series.csv', pd.DataFrame(rows))
    compatible = [row for row in rows if row['header_orientation_compatible']]
    summary = dict(complete=True, local_patient_directories=len(local), existing_split_patients=len(used),
        unused_local_patient_directories=len(unused), unused_metadata_series=len(rows),
        exact_ax_t1_candidate_series=sum(row['exact_current_ax_t1_description'] for row in rows),
        compatible_sample_header_series=len(compatible), full_series_compatibility_established=False,
        development_split_expanded=False, test_ids_used_for_exclusion_only=True,
        test_image_pixels_used=False, unused_image_pixels_loaded=False, new_data_downloaded=False)
    json_write(output / 'complete.json', summary)
    inputs['scripts/v8_generator_dev/audit_unused_local.py'] = digest(__file__)
    json_write(output / 'inputs.json', inputs)
    (output / 'README.md').write_text(
        '# Unused local patient inventory\n\nProducer: `python -m scripts.v8_generator_dev.audit_unused_local`.\n'
        'unused_patients.csv covers every local raw patient outside all existing split memberships. '
        'unused_series.csv retains deduplicated metadata series and any matching-sequence sample-header check. '
        'complete.json reports candidates versus established compatibility; inputs.json hashes metadata/code. '
        'Only test membership IDs were used to exclude patients; no test pixels or acquisition fields '
        'were used for development. An ax-t1 string/sample header is insufficient to establish sequence, '
        'conditioning, full-series geometry or clinical compatibility. No split or dataset is changed.\n', encoding='utf-8')
    print('UNUSED LOCAL INVENTORY', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    audit(parser.parse_args().run_id)
