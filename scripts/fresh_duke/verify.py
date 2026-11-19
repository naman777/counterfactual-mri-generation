"""Independently check the acquired cohort against the frozen patient order and raw files."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from scripts.fresh_duke.acquire import CELLS, ROOT, load_plan, sha, valid_id
from src.data.preprocess import (
    normalize_field_strength, normalize_manufacturer, percentile_clip_and_normalize, read_dicom_image, resize_image,
)


def verify(acquisition_id, verification_id):
    root = ROOT / 'reports/v21_fresh_duke'
    folder = root / valid_id(acquisition_id)
    raw = ROOT / 'data/raw_fresh_duke' / acquisition_id
    lock_path = folder / 'lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    plan, order = load_plan(lock['plan_run'])
    assert plan['order_sha256'] == lock['order_sha256']
    for name, digest in lock['files'].items():
        assert sha(folder / Path(name).name) == digest
    source_manifest = pd.read_csv(root / lock['plan_run'] / 'patient_manifest.csv')
    attempts = pd.read_json(folder / 'attempts.jsonl', lines=True, dtype={'patient_id': str})
    instances = pd.read_csv(folder / 'dicom_inventory.csv')
    slices = pd.read_csv(folder / 'slice_manifest.csv')
    assert attempts.patient_id.is_unique
    assert attempts.status.isin(['accepted', 'excluded']).all()
    assert instances.sop_uid.is_unique and slices.sop_uid.is_unique
    assert set(attempts.patient_id).issubset(set(source_manifest.query("status == 'catalog_eligible'").patient_id))
    assert set(attempts.query("status == 'accepted'").patient_id) == set(instances.patient_id) == set(slices.patient_id)
    assert set(instances.query('selected').sop_uid) == set(slices.sop_uid)
    assert lock['total_slices'] == len(slices)
    bytes_on_disk = sum(path.stat().st_size for path in raw.rglob('*') if path.is_file())
    assert bytes_on_disk == lock['dicom_bytes']

    summary_rows = []
    echo = lock['source_echo_stats']
    for manufacturer, field in CELLS:
        cell_order = order[order.manufacturer.eq(manufacturer) & order.field_strength.eq(field)]
        cell_attempts = attempts[attempts.patient_id.isin(cell_order.PatientID)]
        prefix = cell_order.head(len(cell_attempts))
        assert cell_attempts.patient_id.tolist() == prefix.PatientID.tolist()
        assert cell_attempts.series_uid.tolist() == prefix.SeriesInstanceUID.tolist()
        assert cell_attempts['rank'].tolist() == prefix.within_cell_rank.tolist()
        assert prefix.role.isin(['initial', 'reserve']).all()
        accepted = cell_attempts.query("status == 'accepted'")
        count = len(accepted)
        assert count == lock['counts'][f'{manufacturer} {field}']
        assert count <= plan['target_per_cell']
        if count < plan['target_per_cell']:
            assert len(prefix) == len(cell_order.query("role != 'unreleased'"))
        summary_rows.append({
            'manufacturer': manufacturer, 'field_strength': field, 'patients': count,
            'slices': int(slices.patient_id.isin(accepted.patient_id).sum()),
            'exclusions': len(cell_attempts) - count,
            'reserves_attempted': int(prefix.role.eq('reserve').sum()),
        })

    for patient, group in instances.groupby('patient_id'):
        expected = order.loc[order.PatientID.eq(patient)].iloc[0]
        ordered = group.sort_values('physical_index')
        assert len(ordered) == expected.instanceCount
        assert ordered.physical_index.tolist() == list(range(len(ordered)))
        assert ordered.series_uid.eq(expected.SeriesInstanceUID).all()
        assert ordered.study_uid.eq(expected.StudyInstanceUID).all()
        assert ordered.manufacturer.eq(expected.manufacturer).all()
        assert ordered.field_strength.eq(expected.field_strength).all()
        assert np.all(np.diff(ordered.position_mm) > 0)
        paths = set()
        reference = json.loads(ordered.loc[ordered.dicom_path.idxmin(), 'image_orientation_patient'])
        normal = np.cross(reference[:3], reference[3:])
        normal *= np.sign(normal[2])
        assert normal[2] >= np.cos(np.deg2rad(lock['preprocessing']['max_axial_tilt_deg']))
        manufacturer = {'GE': 'GE MEDICAL SYSTEMS', 'Siemens': 'SIEMENS'}[expected.manufacturer]
        for row in ordered.itertuples(index=False):
            path = ROOT / row.dicom_path
            assert path.resolve().is_relative_to((raw / patient).resolve())
            paths.add(path)
            assert sha(path) == row.dicom_sha256
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            assert str(ds.SOPInstanceUID) == row.sop_uid
            assert str(ds.PatientID) == patient and str(ds.SeriesInstanceUID) == row.series_uid
            assert str(ds.StudyInstanceUID) == row.study_uid
            assert ds.Modality == 'MR' and ds.SeriesDescription.strip().casefold() == 'ax t1'
            assert normalize_manufacturer(ds.Manufacturer) == manufacturer
            assert normalize_field_strength(ds.MagneticFieldStrength) == row.field_strength
            directions = np.asarray(ds.ImageOrientationPatient, dtype=float)
            position = np.asarray(ds.ImagePositionPatient, dtype=float)
            np.testing.assert_array_equal(directions, json.loads(row.image_orientation_patient))
            np.testing.assert_array_equal(position, json.loads(row.image_position_patient))
            assert np.isclose(np.dot(position, normal), row.position_mm, rtol=0, atol=1e-8)
        assert paths == set((raw / patient).rglob('*.dcm'))
        selected = slices.loc[slices.patient_id.eq(patient)].sort_values('physical_index')
        interior = len(ordered) - 2
        k = min(5, interior)
        expected_indices = np.floor((np.arange(k) + .5) * interior / k).astype(int) + 1
        assert selected.physical_index.tolist() == expected_indices.tolist()
        assert selected.sop_uid.tolist() == ordered.iloc[expected_indices].sop_uid.tolist()
        for row in selected.itertuples(index=False):
            dicom, image = ROOT / row.dicom_path, ROOT / row.image_path
            assert image.resolve().is_relative_to((folder / 'images').resolve())
            assert sha(dicom) == row.dicom_sha256 and sha(image) == row.image_sha256
            ds = pydicom.dcmread(dicom, stop_before_pixels=True)
            assert np.isclose(float(ds.EchoTime), row.echo_time, rtol=0, atol=1e-10)
            assert np.isclose((float(ds.EchoTime) - echo['min']) / (echo['max'] - echo['min']),
                              row.echo_time_norm, rtol=0, atol=1e-10)
            replay = resize_image(percentile_clip_and_normalize(read_dicom_image(dicom), 1, 99), 128)
            np.testing.assert_array_equal(np.load(image, allow_pickle=False), replay)
    counts = [row['patients'] for row in summary_rows]
    status = ('target_met' if all(n == plan['target_per_cell'] for n in counts) else
              'minimum_met' if all(n >= plan['minimum_per_cell'] for n in counts) else 'below_minimum')
    assert lock['status'] == status
    summary = {
        'status': lock['status'], 'lock_sha256': sha(lock_path), 'dicom_bytes': bytes_on_disk,
        'accepted_patients': int(instances.patient_id.nunique()), 'verified_dicom_instances': len(instances),
        'reproduced_arrays': len(slices), 'scanner_cells': summary_rows,
        'exclusions_by_reason': attempts.query("status == 'excluded'").reason.value_counts().to_dict()
        if 'reason' in attempts else {},
        'verification_source_sha256': sha(__file__),
    }
    out = root / valid_id(verification_id)
    out.mkdir(exist_ok=False)
    (out / 'verified.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        f'Produced by `python -m scripts.fresh_duke.verify --acquisition-id {acquisition_id} '
        f'--verification-id {verification_id}`. verified.json records the independent rank, eligibility, '
        'scanner balance, DICOM, slice selection, hash and array replay checks.\n', encoding='utf-8',
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition-id', required=True)
    parser.add_argument('--verification-id', required=True)
    args = parser.parse_args()
    verify(args.acquisition_id, args.verification_id)
