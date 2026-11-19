"""Freeze a metadata-only Duke replication cohort and historical precision plan."""

import argparse
import hashlib
import json
from pathlib import Path

import idc_index_data
import numpy as np
import pandas as pd
from scipy.stats import t

from scripts.v4_heldout.audit import ROOT, sha
from scripts.v4_heldout.download import valid_id


CELLS = [(manufacturer, field) for manufacturer in ('GE', 'Siemens') for field in ('1.5T', '3.0T')]
SEED = 'duke-replication-42'
TARGET = 15
RESERVES = 5


def read(relative, **kwargs):
    return pd.read_csv(ROOT / relative, **kwargs)


def prior_patients():
    evidence = []

    def add(ids, reason, source):
        evidence.extend((patient, reason, str(source)) for patient in sorted(set(ids)))

    originals = {}
    for cohort, folder, splits in [
        ('original', 'data/processed/ax_t1_128', ('train', 'val', 'test')),
        ('independent_judge', 'data/processed/ax_t1_128_heldout/preflight_20261016_01', ('train', 'val')),
    ]:
        originals[cohort] = set()
        for split in splits:
            path = f'{folder}/splits/{split}.csv'
            ids = set(read(path, usecols=['patient_id']).patient_id)
            if originals[cohort] & ids:
                raise ValueError(f'Overlapping historical splits: {path}')
            originals[cohort].update(ids)
            add(ids, f'{cohort}_{split}', path)
    if len(originals['original']) != 103 or len(originals['independent_judge']) != 64 or (
        originals['original'] & originals['independent_judge']
    ):
        raise ValueError('Historical memberships differ from the audited 167-patient union')

    for path, column, reason in [
        ('data/metadata/dicom_metadata_sample.csv', 'PatientID', 'earlier_dicom_header_sample'),
        ('data/metadata/clean_metadata.csv', 'PatientID', 'historical_local_collection'),
        ('reports/v8_generator_dev/public_metadata_candidates_20261117_01/series_coverage.csv',
         'patient_id', 'targeted_metadata_screen'),
        ('reports/v8_generator_dev/prospective_series_audit_20261117_01/headers.csv',
         'patient_id', 'prospective_series_header_pilot'),
        ('data/processed/ax_t1_128_prospective_pixel_qc_20261117_01/manifest.csv',
         'patient_id', 'prospective_pixel_qc_and_contact_sheet'),
        ('reports/v8_generator_dev/local_alias_header_audit_20261117_01/candidate_compatibility.csv',
         'patient_id', 'local_sequence_compatibility_audit'),
        ('reports/v7_geometry/native_dce_v1_20261116_01/annotation_manifest.csv',
         'patient_id', 'native_dce_annotation_pilot'),
    ]:
        add(read(path, usecols=[column])[column], reason, path)
    for directory in (ROOT / 'data/raw').rglob('Breast_MRI_*'):
        if directory.is_dir():
            add([directory.name], 'present_in_historical_raw_tree', directory.relative_to(ROOT))
    return pd.DataFrame(evidence, columns=['patient_id', 'reason', 'evidence']).drop_duplicates().sort_values(
        ['patient_id', 'reason', 'evidence']
    )


def precision(out):
    old = 'reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv'
    heldout = 'reports/v4_heldout_judge/b2_local_20261016_01/regrade_old/per_slice.csv'
    test = 'data/processed/ax_t1_128/splits/test.csv'
    a = read(old, usecols=['patient_id', 'example_id', 'fs_hit', 'edit_timestep', 'guidance_scale'])
    b = read(heldout).query(
        "set == 'old_fs_t100_w5' and variant == 'cf' and family == 'orientation_robust' and seed == 0"
    )[['patient_id', 'example_id', 'hit']]
    j = a.merge(b, on=['patient_id', 'example_id'], validate='one_to_one')
    if len(j) != len(a) or len(j) != len(b) or not (
        j.edit_timestep.eq(100) & j.guidance_scale.eq(5) & j.fs_hit.isin([0, 1]) & j.hit.isin([0, 1])
    ).all():
        raise ValueError('Historical paired endpoint failed alignment')
    j['difference'] = j.fs_hit.astype(int) - j.hit.astype(int)
    patient = j.groupby('patient_id').difference.agg(['size', 'mean', 'var']).reset_index()
    labels = read(test, usecols=['patient_id', 'manufacturer_label', 'field_strength_label']).drop_duplicates()
    patient = patient.merge(labels, on='patient_id', validate='one_to_one')
    m = patient['size'].clip(upper=5)
    # Sensitivity under random subsampling of up to five saved paired differences.
    patient['five_slice_added_variance'] = patient['var'].fillna(0) * (1 / m - 1 / patient['size'])
    patient.to_csv(out / 'historical_patient_differences.csv', index=False)
    j.to_csv(out / 'historical_pairs.csv', index=False)
    s = float(patient['mean'].std())
    adjusted = float(np.sqrt(s**2 + patient.five_slice_added_variance.mean()))
    stress = 0.25
    rows = []
    for n in range(16, 101, 4):
        multiplier = t.ppf(.975, n - 4) / np.sqrt(n)
        rows.append((n, n // 4, multiplier*s, multiplier*adjusted, multiplier*stress))
    grid = pd.DataFrame(rows, columns=['patients', 'per_cell', 'historical_half_width',
                                       'five_slice_half_width', 'stress_half_width'])
    grid.to_csv(out / 'precision_grid.csv', index=False)
    target = grid.query('per_cell == @TARGET').iloc[0]
    floor = grid.query('stress_half_width <= 0.08').iloc[0]
    if target.five_slice_half_width > .06:
        raise ValueError('The proposed accrual target misses the historical five-slice precision goal')
    result = {
        'paired_slices': len(j), 'historical_patients': len(patient),
        'historical_equal_patient_mean': float(patient['mean'].mean()), 'historical_sd': s,
        'five_slice_sd': adjusted, 'stress_sd': stress, 'target_half_width_goal': .06,
        'floor_stress_half_width_goal': .08, 'target': target.to_dict(), 'minimum': floor.to_dict(),
        'inputs': {p: sha(ROOT / p) for p in [old, heldout, test]},
    }
    (out / 'precision.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def inventory():
    index = Path(idc_index_data.IDC_INDEX_PARQUET_FILEPATH)
    catalog = pd.read_parquet(index, filters=[('collection_id', '==', 'duke_breast_cancer_mri')])
    prior = prior_patients()
    excluded = prior.groupby('patient_id').reason.agg(lambda s: ';'.join(sorted(set(s))))
    frozen_labels_path = 'reports/v4_heldout_judge/preflight_20261016_01/catalog_candidates.csv'
    labels = read(frozen_labels_path, usecols=['PatientID', 'planning_field_strength_label',
                                               'planning_manufacturer_label']).drop_duplicates()
    if not labels.PatientID.is_unique:
        raise ValueError('Ambiguous patient-level labels in the saved catalogue audit')
    # Resolve catalogue aliases against all modalities and against historical UIDs.
    known = catalog.loc[catalog.PatientID.isin(excluded.index)]
    historical = read('data/metadata/clean_metadata.csv', usecols=['StudyInstanceUID', 'SeriesInstanceUID'])
    aliases = set()
    for uid in ['StudyInstanceUID', 'SeriesInstanceUID']:
        duplicated = catalog.groupby(uid).PatientID.nunique()
        suspect = set(duplicated[duplicated > 1].index) | set(known[uid].dropna()) | set(historical[uid].dropna())
        aliases.update(catalog.loc[catalog[uid].isin(suspect), 'PatientID'])
    exact = catalog[catalog.Modality.eq('MR') & catalog.SeriesDescription.fillna('').str.strip().str.lower().eq('ax t1')].copy()
    exact = exact.merge(labels, on='PatientID', how='left', validate='many_to_one')
    exact['manufacturer'] = exact.Manufacturer.map({'GE MEDICAL SYSTEMS': 'GE', 'SIEMENS': 'Siemens'})
    exact['field_strength'] = exact.planning_field_strength_label.map({0: '1.5T', 1: '3.0T'})
    exact['planning_match'] = exact.manufacturer.eq(exact.planning_manufacturer_label.map({0: 'GE', 1: 'Siemens'}))
    exact['usable_catalog_entry'] = (exact.planning_match & exact.field_strength.notna()
                                     & pd.to_numeric(exact.series_size_MB).gt(0)
                                     & pd.to_numeric(exact.instanceCount).ge(3))
    rows, candidates = [], []
    for patient in sorted(set(catalog.PatientID.dropna()) | set(excluded.index)):
        series = exact[exact.PatientID.eq(patient)].sort_values('SeriesInstanceUID')
        good = series[series.usable_catalog_entry]
        reason = excluded.get(patient)
        if reason is None:
            if patient in aliases:
                reason = 'shared_historical_or_cross_patient_catalog_uid'
            elif series.empty:
                reason = 'no_exact_ax_t1_catalog_series'
            elif good.empty or not series.usable_catalog_entry.all():
                reason = 'missing_or_inconsistent_catalog_labels_or_series_metadata'
            elif len(good[['manufacturer', 'field_strength']].drop_duplicates()) != 1:
                reason = 'ambiguous_scanner_cell'
        row = {
            'patient_id': patient, 'status': 'catalog_eligible' if reason is None else 'excluded',
            'reason': reason or '', 'exact_ax_t1_series': len(series),
            'manufacturer': '', 'field_strength': '', 'selected_series_uid': '',
        }
        if len(series) and series.manufacturer.nunique() <= 1 and series.field_strength.nunique() <= 1:
            row.update(manufacturer=series.iloc[0].manufacturer, field_strength=series.iloc[0].field_strength)
        if reason is None:
            chosen = good.iloc[0].to_dict()
            chosen['rank_hash'] = hashlib.sha256(f'{SEED}:{patient}'.encode()).hexdigest()
            candidates.append(chosen)
            row['selected_series_uid'] = chosen['SeriesInstanceUID']
        rows.append(row)
    columns = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SeriesDescription',
               'manufacturer', 'field_strength', 'instanceCount', 'series_size_MB', 'rank_hash']
    order = pd.DataFrame(candidates)[columns].sort_values(['manufacturer', 'field_strength', 'rank_hash'])
    order['within_cell_rank'] = order.groupby(['manufacturer', 'field_strength']).cumcount() + 1
    order['role'] = np.select([order.within_cell_rank.le(TARGET), order.within_cell_rank.le(TARGET + RESERVES)],
                              ['initial', 'reserve'], default='unreleased')
    return pd.DataFrame(rows), prior, exact, order, index, frozen_labels_path


def main(run_id):
    out = ROOT / 'reports/v21_fresh_duke' / valid_id(run_id)
    out.mkdir(parents=True, exist_ok=False)
    size = precision(out)
    manifest, prior, series, order, index, labels = inventory()
    manifest.to_csv(out / 'patient_manifest.csv', index=False)
    prior.to_csv(out / 'exclusion_evidence.csv', index=False)
    series.to_csv(out / 'exact_ax_t1_series.csv', index=False)
    order.to_csv(out / 'seeded_series_order.csv', index=False)
    counts = order.groupby(['manufacturer', 'field_strength', 'role']).agg(
        patients=('PatientID', 'size'), instances=('instanceCount', 'sum'),
        catalogue_MB=('series_size_MB', 'sum')
    ).reset_index()
    counts.to_csv(out / 'counts.csv', index=False)
    released = order[order.role.ne('unreleased')]
    initial = order[order.role.eq('initial')]
    if set(map(tuple, initial[['manufacturer', 'field_strength']].drop_duplicates().values)) != set(CELLS) or len(initial) != 4*TARGET:
        raise ValueError('Catalogue cannot meet the prespecified balanced target')
    maps = ROOT / 'data/processed/ax_t1_128/label_maps'
    frozen = [Path(__file__), Path(__file__).with_name('acquire.py'),
              ROOT / 'src/data/preprocess.py', *sorted(maps.glob('*.json'))]
    summary = {
        'catalogue_version': idc_index_data.__version__, 'catalogue_sha256': sha(index),
        'planning_label_source': labels, 'planning_label_sha256': sha(ROOT / labels),
        'prior_patients': prior.patient_id.nunique(), 'total_manifest_patients': len(manifest),
        'catalog_eligible_patients': len(order), 'seed': SEED,
        'initial_patients': len(initial), 'initial_instances': int(initial.instanceCount.sum()),
        'initial_catalogue_GB': float(initial.series_size_MB.sum() / 1000),
        'released_patients_including_reserves': len(released),
        'released_catalogue_GB': float(released.series_size_MB.sum() / 1000),
        'minimum_per_cell': int(size['minimum']['per_cell']), 'target_per_cell': TARGET,
        'order_sha256': sha(out / 'seeded_series_order.csv'),
        'patient_manifest_sha256': sha(out / 'patient_manifest.csv'),
        'frozen_code_and_maps': {p.relative_to(ROOT).as_posix(): sha(p) for p in frozen},
    }
    (out / 'plan.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
