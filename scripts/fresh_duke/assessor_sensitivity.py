"""Test recognition of known-label native replacements and graded image mixtures."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fresh_duke import replication as r
from scripts.fresh_duke.assessor_controls import check
from scripts.fresh_duke.image_diagnostics import RUN
from scripts.fresh_duke.paired_stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, PRIMARY_JUDGES
from scripts.v4_convergence.core import probabilities


REPORT = r.ROOT / 'reports/v27_assessor_sensitivity'
FRACTIONS = (0., .25, .5, .75, 1.)


def dyads(cohort):
    """Match patients within manufacturer and pair slices by within-series rank."""
    table = cohort[['slice_id', 'patient_id', 'mfg', 'fs', 'physical_index']].copy()
    table['row'] = table.index
    ordered = table.sort_values(['patient_id', 'physical_index'])
    ordered['rank'] = ordered.groupby('patient_id').cumcount()
    result = []
    for mfg, group in ordered.groupby('mfg', sort=True):
        by_fs = [group[group.fs.eq(fs)].patient_id.unique() for fs in (0, 1)]
        by_fs = [sorted(ids, key=lambda pid: hashlib.sha256(f'{BOOTSTRAP_SEED}:{pid}'.encode()).digest())
                 for ids in by_fs]
        for index, (low, high) in enumerate(zip(*by_fs, strict=True)):
            a, b = [group[group.patient_id.eq(pid)].sort_values('rank') for pid in (low, high)]
            np.testing.assert_array_equal(a['rank'], b['rank'])
            a, b = a.reset_index(drop=True), b.reset_index(drop=True)
            for source, target in ((a, b), (b, a)):
                result.append(source.assign(dyad=f'{mfg}:{index:02}', target_patient=target.patient_id,
                                            target_slice=target.slice_id, donor=target.row))
    pairs = pd.concat(result, ignore_index=True)
    if len(pairs) != len(cohort) or not pairs.slice_id.is_unique or not pairs.target_slice.is_unique:
        raise ValueError('Each input slice must occur once as source and once as donor')
    if not pairs.groupby('dyad').size().eq(10).all() or not pairs.groupby('mfg').dyad.nunique().eq(15).all():
        raise ValueError('Expected 15 disjoint five-slice patient pairs per manufacturer')
    np.testing.assert_array_equal(pairs.set_index('row').loc[pairs.donor, 'donor'], pairs.row)
    return pairs


def grade(cohort, pairs, protocol, provenance):
    native = np.stack([r.image_array(r.ROOT / row.image_path, row.image_sha256) for row in cohort.itertuples()])
    saved_path = RUN / 'real/per_slice.csv'
    check(saved_path, r.load_json(RUN / 'real/complete.json')['per_slice_sha256'], provenance)
    saved = pd.read_csv(saved_path).query(
        "axis == 'field_strength' and orientation == 'native'").set_index(['judge', 'slice_id'])
    source, donor = native[pairs.row], native[pairs.donor]
    targets = 1 - pairs.fs.to_numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for judge in PRIMARY_JUDGES:
        path = protocol['judges'][judge]
        check(r.ROOT / path, protocol['files_sha256'][path], provenance)
        model = r.load_assessor(r.ROOT / path, judge, device)
        original = judge == PRIMARY_JUDGES[0]
        native_score = probabilities(model, native, original=original, batch=32)['fs']
        prior = saved.loc[judge].loc[cohort.slice_id]
        np.testing.assert_array_equal(native_score.argmax(axis=1), prior.prediction)
        np.testing.assert_allclose(native_score, prior[['p0', 'p1']], atol=2e-5, rtol=0)
        np.testing.assert_array_equal(prior.truth, cohort.fs)
        for fraction in FRACTIONS:
            if fraction == 0:
                score = native_score[pairs.row]
            elif fraction == 1:
                score = native_score[pairs.donor]
            else:
                mixture = (1 - fraction) * source + fraction * donor
                score = probabilities(model, mixture, original=original, batch=32)['fs']
            hit = score.argmax(axis=1) == targets
            rows.append(pairs.assign(judge=judge, fraction=fraction, target_fs=targets,
                                     target_probability=score[np.arange(len(score)), targets], hit=hit))
    return pd.concat(rows, ignore_index=True)


def dyad_estimate(table, value):
    dyad = table.groupby(['mfg', 'dyad'])[value].mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = sum(rng.choice(values.to_numpy(), size=(BOOTSTRAP_DRAWS, len(values))).sum(axis=1)
               for _, values in dyad.groupby(level='mfg'))
    low, high = np.quantile(boot / len(dyad), [.025, .975])
    return dict(estimate=float(dyad.mean()), lo95=float(low), hi95=float(high),
                dyads=len(dyad), slice_pairs=len(table))


def summarize(rows):
    results = []
    measures = ['hit', 'target_probability']
    for judge, group in rows.groupby('judge'):
        baseline = group[group.fraction.eq(0)].set_index('slice_id')
        for fraction, trial in group.groupby('fraction'):
            trial = trial.copy()
            for measure in measures:
                trial[f'{measure}_change'] = (trial[measure].to_numpy(dtype=float)
                                               - baseline.loc[trial.slice_id, measure].to_numpy(dtype=float))
            for target, subset in [('pooled', trial), *trial.groupby('target_fs')]:
                for value in (*measures, *(f'{measure}_change' for measure in measures)):
                    results.append(dict(judge=judge, fraction=fraction, target_fs=target, measure=value,
                                        **dyad_estimate(subset, value)))
    return pd.DataFrame(results)


def main(out):
    inputs = {}
    protocol_path = RUN / 'protocol.json'
    check(protocol_path, r.load_json(RUN / 'complete.json')['protocol_sha256'], inputs)
    protocol = r.load_json(protocol_path)
    check(RUN / 'inputs.csv', protocol['input_sha256'], inputs)
    cohort = pd.read_csv(RUN / 'inputs.csv')
    pairs = dyads(cohort)
    predictions = grade(cohort, pairs, protocol, inputs)
    out.mkdir(parents=True, exist_ok=False)
    r.save_csv(out / 'pairs.csv', pairs)
    r.save_csv(out / 'predictions.csv', predictions)
    r.save_csv(out / 'sensitivity.csv', summarize(predictions))
    r.json_write(out / 'protocol.json', {'inputs': inputs, 'script_sha256': r.digest(Path(__file__)),
                                        'fractions': FRACTIONS, 'seed': BOOTSTRAP_SEED,
                                        'draws': BOOTSTRAP_DRAWS})
    (out / 'README.md').write_text(
        '# Known-label field-strength sensitivity control\n\n'
        'All outputs come from `python -m scripts.fresh_duke.assessor_sensitivity --out <new-folder>`. '
        '`pairs.csv` matches previously unused patients across field strength within manufacturer by stable hashed '
        'order, then matches slices by rank within each patient. Each pair is reversible. '
        '`predictions.csv` records the two original assessors’ probabilities and hard calls at each donor fraction. '
        '`sensitivity.csv` reports absolute and source-relative responses, with bootstrap intervals from resampling '
        'patient dyads within manufacturer. `protocol.json` records the fractions and input hashes.\n\n'
        'The native endpoints have known labels and are checked against the frozen replication grades. '
        'Intermediate mixtures are synthetic dose probes and have no known acquisition label. '
        'Reciprocal mixtures at complementary fractions share pixels but have opposite requested labels. '
        'The pooled 50% mixture hit rate is therefore fixed at 50%, and complementary fractions are dependent. '
        'The donor has different anatomy, and mixing changes many features at once. This control measures '
        'assessor response to target images and mixtures, retaining the need for paired acquisition ground truth.\n',
        encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPORT / 'control_20261119_01')
    main(parser.parse_args().out)
