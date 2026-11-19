"""Audit saved field-strength decisions without rerunning the generator or judges."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fresh_duke.paired_stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, PRIMARY_JUDGES, patient_estimate
from scripts.v4_heldout.audit import ROOT, sha


REPORT = ROOT / 'reports'
OUTPUT = REPORT / 'v23_replication_analysis'
RUN = REPORT / 'v22_duke_replication/replication_20261118_02'
HELD = REPORT / 'v4_heldout_judge/b2_local_20261016_01'
JOB = REPORT / 'v4_diffusion/b1_20261016_01/remote/full/test/main_field_strength_t100_w5_s0'
KEYS = ['slice_id', 'patient_id']
SUBSETS = [('all', None), ('1.5T_to_3.0T', 0), ('3.0T_to_1.5T', 1)]


def load(path, hashes, expected=None):
    digest = sha(path)
    if expected is not None and expected != digest:
        raise ValueError(f'Locked source hash mismatch for {path}')
    hashes[path.relative_to(ROOT).as_posix()] = digest
    return pd.read_csv(path)


def historical(hashes):
    anchor = load(RUN / 'historical/paired_slices.csv', hashes)
    identity = anchor[KEYS + ['mfg', 'fs', 'target_label']]
    for epoch, archived, key in [(10, 'old', 'old_fs_t100_w5'), (110, 'b3', JOB.name)]:
        held = load(HELD / f'regrade_{archived}/per_slice.csv', hashes)
        held = held[held['set'].eq(key) & held.variant.eq('cf')].rename(columns={'example_id': 'slice_id'})
        frames = []
        for (family, seed), rows in held.groupby(['family', 'seed']):
            joined = identity.merge(rows[KEYS + ['hit', 'fs_source', 'fs_target', 'mfg_source']],
                                    on=KEYS, how='outer', validate='one_to_one')
            for a, b in [('fs', 'fs_source'), ('target_label', 'fs_target'), ('mfg', 'mfg_source')]:
                if not joined[a].eq(joined[b]).all():
                    raise ValueError(f'Archived held-out labels disagree for {epoch}, {family}, {seed}')
            frames.append(joined[KEYS + ['hit']].assign(judge=f'heldout_{family}_seed{seed}'))
        if epoch == 10:
            frames.append(anchor[KEYS + ['hit_original']].rename(columns={'hit_original': 'hit'})
                          .assign(judge=PRIMARY_JUDGES[0]))
            for tag, family in [('', 'robust'), ('_plain', 'flip_only')]:
                archive = load(REPORT / f'v2_quickwins/regrade{tag}/regrade_per_example.csv', hashes)
                subset = archive[archive['set'].eq('diffusion_fs_t100_w5')].rename(columns={'example_id': 'slice_id'})
                frames.append(subset[KEYS + ['hit']].assign(
                    judge=subset.judge_seed.map(lambda seed: f'training_{family}_seed{seed}')))
        else:
            ck = json.loads((JOB / 'job.json').read_text())
            hashes[(JOB / 'job.json').relative_to(ROOT).as_posix()] = sha(JOB / 'job.json')
            checkpoint = ROOT / 'checkpoints/diffusion/ax_t1_128_v4/b1_20261016_01/selected_epoch_0110.pt'
            if (ck['t'], ck['w'], ck['offset'], ck['checkpoint_sha256']) != (100, 5, 0, sha(checkpoint)):
                raise ValueError('Historical epoch-110 checkpoint or operating point changed')
            metrics = load(JOB / 'metrics.csv', hashes).rename(columns={'example_id': 'slice_id', 'fs_hit': 'hit'})
            metrics = metrics[metrics.variant.eq('cf')].copy()
            for name in metrics.judge.unique():
                group = metrics[metrics.judge.eq(name)]
                checked = identity.merge(group[KEYS + ['fs_source', 'fs_target', 'mfg_source']],
                                         on=KEYS, how='outer', validate='one_to_one')
                if not (checked.fs.eq(checked.fs_source) & checked.target_label.eq(checked.fs_target)
                        & checked.mfg.eq(checked.mfg_source)).all():
                    raise ValueError(f'Mismatched historical epoch-110 labels for {name}')
            rename = {f'{old}{seed}': f'training_{new}_seed{seed}'
                      for old, new in [('robust', 'robust'), ('plain', 'flip_only')] for seed in range(3)}
            metrics['judge'] = metrics.judge.replace(rename)
            frames.append(metrics[KEYS + ['judge', 'hit']])
        table = pd.concat(frames).merge(identity, on=KEYS, how='outer', validate='many_to_one')
        if epoch == 10:
            replay = table[table.judge.eq(PRIMARY_JUDGES[1])].merge(anchor[KEYS + ['hit_heldout']], on=KEYS)
            np.testing.assert_array_equal(replay.hit, replay.hit_heldout)
        yield epoch, table


def fresh(hashes):
    for epoch in [10, 110]:
        folder = RUN / f'epoch{epoch}_field_strength'
        completion = folder / 'complete.json'
        result = json.loads(completion.read_text())
        hashes[completion.relative_to(ROOT).as_posix()] = sha(completion)
        if result['status'] != 'complete' or result['failed_arm_slices'] != 0:
            raise ValueError(f'Incomplete fresh epoch {epoch}')
        grades = load(folder / 'per_slice.csv', hashes, result['per_slice_sha256'])
        yield epoch, grades[grades.arm.eq('target')]


def intervals_by_direction(group, scanner_strata):
    """Descriptive difference between the two direction-specific patient means."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {}
    rates = {}
    for source, rows in group.groupby('fs'):
        means = rows.groupby(['patient_id', 'mfg']).difference.mean()
        groups = means.groupby(level='mfg') if scanner_strata else [(None, means)]
        bootstrap = np.zeros(BOOTSTRAP_DRAWS)
        for _, group_means in groups:
            values = group_means.to_numpy()
            bootstrap += rng.choice(values, (BOOTSTRAP_DRAWS, len(values))).sum(axis=1)
        draws[source] = bootstrap / len(means)
        rates[source] = means.mean()
    lo, hi = np.quantile(draws[0] - draws[1], [.025, .975])
    return dict(estimate=rates[0] - rates[1], lo95=lo, hi95=hi)


def analyse(records, cohort, epoch):
    n, p = (100, 16) if cohort == 'historical' else (300, 60)
    if records.judge.nunique() != 13 or len(records) != n * 13 or records.patient_id.nunique() != p:
        raise ValueError(f'Incomplete judge or case coverage for {cohort}, epoch {epoch}')
    if not records.hit.isin([0, 1]).all() or not (records.fs + records.target_label).eq(1).all():
        raise ValueError('Unexpected labels or hard decisions')
    wide = records.pivot(index=KEYS + ['mfg', 'fs'], columns='judge', values='hit').reset_index()
    if len(wide) != n or wide.isna().any().any() or not wide.slice_id.is_unique:
        raise ValueError(f'Target grades do not pair one-to-one for {cohort}, epoch {epoch}')
    if cohort == 'fresh' and (not wide.groupby('patient_id').size().eq(5).all()
                             or not wide.groupby(['mfg', 'fs']).patient_id.nunique().eq(15).all()):
        raise ValueError('Fresh design is no longer balanced')
    scanner_strata = cohort == 'fresh'
    overall, omissions, directions = [], [], []
    for judge in sorted(set(records.judge) - {PRIMARY_JUDGES[0]}):
        pairs = wide[KEYS + ['mfg', 'fs', PRIMARY_JUDGES[0], judge]].rename(
            columns={PRIMARY_JUDGES[0]: 'original', judge: 'hit'})
        pairs['difference'] = pairs.original.astype(int) - pairs.hit.astype(int)
        context = dict(cohort=cohort, epoch=epoch, judge=judge)
        directions.append({**context, **intervals_by_direction(pairs, scanner_strata)})
        for name, label in SUBSETS:
            sample = pairs if label is None else pairs[pairs.fs.eq(label)]
            result = patient_estimate(sample, 'difference', scanner_strata)
            descriptor = {**context, 'direction': name}
            cells = {key: int((sample.original.eq(a) & sample.hit.eq(b)).sum()) for key, a, b in [
                ('both', 1, 1), ('original_only', 1, 0), ('other_only', 0, 1), ('neither', 0, 0)]}
            assert sum(cells.values()) == len(sample)
            per_patient = sample.groupby('patient_id').difference.mean()
            # Recompute the equal-patient estimand directly on each retained set.
            loo = {patient: per_patient.drop(patient).mean() for patient in per_patient.index}
            if not np.isfinite(list(loo.values())).all():
                raise ValueError('Leave-one-out sensitivity has an empty retained set')
            omissions.extend({**descriptor, 'omitted_patient': patient, 'estimate': value,
                              'shift': value - result['estimate']} for patient, value in loo.items())
            rates = {}
            for name_, column in [('original', 'original'), ('other', 'hit')]:
                rate = patient_estimate(sample, column, scanner_strata)
                rates.update({f'{name_}_{k}': rate[k] for k in ['estimate', 'lo95', 'hi95']})
            overall.append({**descriptor, **cells, **result, **rates, 'slice_weighted': sample.difference.mean(),
                            'loo_min': min(loo.values()), 'loo_max': max(loo.values())})
    return overall, omissions, directions


def main(output):
    hashes = {}
    summaries, omissions, contrasts = [], [], []
    for cohort, stream in [('historical', historical(hashes)), ('fresh', fresh(hashes))]:
        for epoch, records in stream:
            a, b, c = analyse(records, cohort, epoch)
            summaries.extend(a)
            omissions.extend(b)
            contrasts.extend(c)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(summaries)
    for cohort, epoch, recorded in [
        ('historical', 10, RUN / 'historical/primary.json'),
        ('fresh', 10, RUN / 'epoch10_field_strength/primary.json'),
        ('fresh', 110, RUN / 'epoch110_field_strength/primary.json'),
    ]:
        old = json.loads(recorded.read_text())
        hashes[recorded.relative_to(ROOT).as_posix()] = sha(recorded)
        for direction, _ in SUBSETS:
            row = frame.query('cohort == @cohort and epoch == @epoch and judge == @PRIMARY_JUDGES[1] and direction == @direction').iloc[0]
            expected = old[direction]
            np.testing.assert_allclose(row[['estimate', 'lo95', 'hi95']].astype(float),
                                       [expected[key] for key in ['estimate', 'lo95', 'hi95']], atol=1e-12)
            np.testing.assert_equal(row[['both', 'original_only', 'other_only', 'neither']].tolist(),
                                    list(expected['agreement_counts'].values()))
    frame.to_csv(out / 'paired_summary.csv', index=False)
    pd.DataFrame(omissions).to_csv(out / 'leave_one_out.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'direction_contrasts.csv', index=False)
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
    stats = Path(__file__).with_name('paired_stats.py')
    hashes[stats.relative_to(ROOT).as_posix()] = sha(stats)
    (out / 'inputs.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        '# Paired field-strength analysis\n\n'
        'Produced by `python -m scripts.fresh_duke.replication_analysis`. inputs.json fingerprints '
        'the archived grades, fresh grades, reproduced estimates and analysis code. Existing locked estimates '
        'and agreement counts were checked from the saved hard decisions. No generation or grading was rerun.\n\n'
        'paired_summary.csv gives the four agreement counts, slice and patient denominators, patient-weighted '
        'hit rates and differences with 95% percentile intervals, slice-weighted differences and leave-one-out '
        'ranges. Each comparison uses the original predictor and one named judge on identical saved targets. '
        'leave_one_out.csv gives all patient omissions and shifts. direction_contrasts.csv compares the '
        '1.5-to-3.0 T gap with the reverse-direction gap, with a descriptive 95% interval. Historical resampling '
        'is unstratified and fresh resampling preserves scanner cells. For leave-one-out estimates we average '
        'the retained patients equally. Direction contrasts, non-primary judges and epoch 110 are descriptive.\n',
        encoding='utf-8',
    )
    print(frame[frame.judge.eq(PRIMARY_JUDGES[1])][
        ['cohort', 'epoch', 'direction', 'patients', 'slices', 'both', 'original_only', 'other_only', 'neither',
         'estimate', 'lo95', 'hi95', 'loo_min', 'loo_max']].to_string(index=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=OUTPUT / 'analysis_20261119_01')
    main(cli.parse_args().output)
