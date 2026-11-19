"""Patient-level paired analysis shared by the historical and replication tables."""

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 10000
PRIMARY_JUDGES = ('original_predictor', 'heldout_orientation_robust_seed0')


def patient_estimate(rows, value, stratified):
    patient = rows.groupby(['patient_id', 'mfg', 'fs'], as_index=False)[value].mean()
    if patient.empty:
        return {'slices': 0, 'patients': 0, 'estimate': None, 'lo95': None, 'hi95': None}
    if not patient.patient_id.is_unique:
        raise ValueError('Conflicting scanner strata for a patient')
    groups = patient.groupby(['mfg', 'fs'])[value] if stratified else [(None, patient[value])]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.zeros(BOOTSTRAP_DRAWS)
    for _, group in groups:
        values = group.to_numpy(dtype=float)
        chosen = rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        samples += values[chosen].sum(axis=1)
    lo, hi = np.quantile(samples / len(patient), [.025, .975])
    return {'slices': len(rows), 'patients': len(patient), 'estimate': float(patient[value].mean()),
            'lo95': float(lo), 'hi95': float(hi)}


def primary_pairs(records):
    columns = ['slice_id', 'patient_id', 'mfg', 'fs', 'target_label', 'judge', 'hit', 'native_source_correct']
    a, b = (records.loc[records.judge.eq(j), columns].drop(columns='judge') for j in PRIMARY_JUDGES)
    keys = columns[:5]
    patients = a.merge(b, on=keys, suffixes=('_original', '_heldout'), validate='one_to_one')
    patients['difference'] = patients.hit_original.astype(int) - patients.hit_heldout.astype(int)
    return patients


def primary_report(rows, stratified):
    estimates = {}
    for subset, group in [
        ('all', rows),
        ('source_correct_both', rows[rows.native_source_correct_original & rows.native_source_correct_heldout]),
        ('1.5T_to_3.0T', rows[rows.fs.eq(0)]), ('3.0T_to_1.5T', rows[rows.fs.eq(1)]),
    ]:
        result = patient_estimate(group, 'difference', stratified)
        result['original_predictor'] = patient_estimate(group, 'hit_original', stratified)
        result['heldout_orientation_robust_seed0'] = patient_estimate(group, 'hit_heldout', stratified)
        result['agreement_counts'] = {
            name: int((group.hit_original.eq(a) & group.hit_heldout.eq(b)).sum())
            for name, a, b in [('both_hit', 1, 1), ('original_only', 1, 0),
                               ('heldout_only', 0, 1), ('neither', 0, 0)]
        }
        result['slice_weighted_difference'] = float(group.difference.mean()) if len(group) else None
        estimates[subset] = result
    patients = rows.groupby(['patient_id', 'mfg', 'fs'], as_index=False).agg(
        slices=('slice_id', 'size'), hit_fraction_original=('hit_original', 'mean'),
        hit_fraction_heldout=('hit_heldout', 'mean'), difference=('difference', 'mean'),
    )
    if len(patients) > 1:
        patients['leave_one_out_difference'] = (patients.difference.sum() - patients.difference) / (len(patients) - 1)
    return estimates, patients


def summarise(records, metric, keys):
    rows = []
    for group_keys, group in records.groupby(keys, sort=False):
        info = dict(zip(keys, group_keys, strict=True))
        rows.append({**info, 'subset': 'all', **patient_estimate(group, metric, True)})
        label = 'truth' if metric == 'correct' else 'target_label'
        for label_value, subset in group.groupby(label):
            rows.append({**info, 'subset': f'{label}_{int(label_value)}', **patient_estimate(subset, metric, True)})
    return pd.DataFrame(rows)
