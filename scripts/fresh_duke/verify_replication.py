"""Check the frozen replication logs, coverage, primary probabilities and paired inference."""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from scripts.fresh_duke import replication as r
from scripts.fresh_duke.paired_stats import PRIMARY_JUDGES


def probability_check(grades):
    np.testing.assert_allclose(grades.p0 + grades.p1, 1, rtol=0, atol=4e-7)
    assert np.isfinite(grades[['p0', 'p1']]).all().all()
    np.testing.assert_array_equal(grades.prediction, (grades.p1 > grades.p0).astype(int))


def real_check(run, protocol, cohort):
    folder = run / 'real'
    result = r.load_json(folder / 'complete.json')
    file = folder / 'per_slice.csv'
    assert result['per_slice_sha256'] == r.digest(file)
    table = pd.read_csv(file)
    assert len(table) == result['prediction_rows'] == len(cohort) * len(protocol['judges']) * 4
    assert not table.duplicated(['slice_id', 'judge', 'axis', 'orientation']).any()
    assert set(table.slice_id) == set(cohort.slice_id)
    assert set(table.judge) == set(protocol['judges'])
    assert set(table.orientation) == {'native', 'vertical_flip'} and set(table.axis) == set(r.AXES)
    probability_check(table)
    np.testing.assert_array_equal(table.correct, table.prediction.eq(table.truth))
    mapping = cohort.set_index('slice_id')
    for axis, label in r.AXES.items():
        sub = table[table.axis.eq(axis)]
        np.testing.assert_array_equal(sub.truth, mapping.loc[sub.slice_id, label])
    return len(table)


def read_images(folder, static, n):
    events = [json.loads(line) for line in (folder / 'generation.jsonl').read_text(encoding='utf-8').splitlines()]
    assets = {}
    for event in events:
        if event['status'] != 'saved':
            continue
        file = folder / 'chunks' / f"{event['offset']:04d}.npz"
        assert r.digest(file) == event['sha256']
        with np.load(file, allow_pickle=False) as pack:
            np.testing.assert_array_equal(pack['offsets'], np.arange(event['offset'], event['stop']))
            assert set(pack.files) == set(event['arms']) | {'offsets'}
            for arm in event['arms']:
                assets.setdefault(arm, []).append((pack['offsets'], pack[arm]))
    images = dict(static)
    for arm, chunks in assets.items():
        np.testing.assert_array_equal(np.concatenate([indices for indices, _ in chunks]), np.arange(n))
        images[arm] = np.concatenate([img for _, img in chunks])
    for array in images.values():
        assert array.shape == (n, 128, 128) and array.dtype == np.float32 and np.isfinite(array).all()
        assert array.min() >= 0 and array.max() <= 1
    return images, events


def independent_primary(folder, grades, protocol):
    saved = r.load_json(folder / 'primary.json')['all']
    targets = grades.query("arm == 'target'").pivot(index=['slice_id', 'patient_id', 'mfg', 'fs'],
                                                   columns='judge', values='hit').reset_index()
    original, heldout = (targets[j].astype(int) for j in PRIMARY_JUDGES)
    hits = {(a, b): int((original.eq(a) & heldout.eq(b)).sum()) for a in (0, 1) for b in (0, 1)}
    assert saved['agreement_counts'] == dict(both_hit=hits[1, 1], original_only=hits[1, 0],
                                           heldout_only=hits[0, 1], neither=hits[0, 0])
    observed = targets.assign(difference=original - heldout)
    patient = observed.groupby(['mfg', 'fs', 'patient_id']).difference.mean()
    rng = np.random.default_rng(protocol['resampling']['seed'])
    size = protocol['resampling']['draws']
    samples = np.zeros(size)
    for _, v in patient.groupby(level=[0, 1]):
        samples += rng.choice(v.to_numpy(), (size, len(v)), replace=True).sum(1)
    lo, hi = np.percentile(samples / len(patient), [2.5, 97.5])
    np.testing.assert_allclose([saved['estimate'], saved['lo95'], saved['hi95']], [patient.mean(), lo, hi], atol=1e-12)
    assert saved['slices'] == len(targets) and saved['patients'] == len(patient)
    return saved


def phase_check(folder, protocol, cohort, static, axis):
    result = r.load_json(folder / 'complete.json')
    if result['status'] != 'complete':
        raise ValueError(f'Cannot certify incomplete outcome: {folder.name}')
    assert r.digest(folder / 'generation.jsonl') == result['generation_log_sha256']
    assert r.digest(folder / 'per_slice.csv') == result['per_slice_sha256']
    images, events = read_images(folder, static, len(cohort))
    grades = pd.read_csv(folder / 'per_slice.csv')
    assert result['predictions'] == len(grades) == len(images) * len(protocol['judges']) * len(cohort)
    assert set(grades.judge) == set(protocol['judges']) and set(grades.arm) == set(images)
    assert not grades.duplicated(['slice_id', 'judge', 'arm']).any()
    assert set(grades.slice_id) == set(cohort.slice_id)
    assert result['paired_target_slices'] == len(cohort) and result['failed_arm_slices'] == 0
    label = r.AXES[axis]
    mapping = cohort.set_index('slice_id')
    probability_check(grades)
    np.testing.assert_array_equal(grades.source_label, mapping.loc[grades.slice_id, label])
    np.testing.assert_array_equal(grades.target_label, 1 - grades.source_label)
    np.testing.assert_array_equal(grades.hit, grades.prediction.eq(grades.target_label))
    np.testing.assert_array_equal(grades.source_agreement, grades.prediction.eq(grades.source_label))
    controls = pd.read_csv(folder / 'paired_controls.csv')
    assert len(controls) == (len(images) - 1) * len(protocol['judges']) * len(cohort)
    assert not controls.duplicated(['slice_id', 'judge', 'control']).any()
    np.testing.assert_array_equal(controls.difference, controls.hit_target.astype(int) - controls.hit_control.astype(int))
    max_abs = 0.
    for judge in PRIMARY_JUDGES:
        model = r.load_assessor(r.ROOT / protocol['judges'][judge], judge, 'cuda')
        for arm, array in images.items():
            recorded = grades[(grades.judge == judge) & (grades.arm == arm)].set_index('slice_id').loc[cohort.slice_id]
            probabilities = r.probabilities(model, array, original=judge == PRIMARY_JUDGES[0], batch=32)[label]
            residual = np.abs(recorded[['p0', 'p1']].to_numpy() - probabilities)
            max_abs = max(max_abs, float(residual.max()))
            np.testing.assert_allclose(recorded[['p0', 'p1']], probabilities, rtol=0, atol=2e-6)
            np.testing.assert_array_equal(recorded.prediction, probabilities.argmax(1))
        del model
    summary = {'prediction_rows': len(grades), 'control_pairs': len(controls),
               'generation_retry_events': sum(event['status'] == 'failed' for event in events),
               'max_primary_probability_error': max_abs, 'arm_count': len(images)}
    if axis == 'field_strength':
        summary['primary'] = independent_primary(folder, grades, protocol)
    return summary


def main(run_id, verification_id):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    run = r.REPORT / r.valid_id(run_id)
    out = r.REPORT / r.valid_id(verification_id)
    protocol = r.read_freeze(run)
    complete = r.load_json(run / 'complete.json')
    assert r.digest(run / 'protocol.json') == complete['protocol_sha256']
    assert r.digest(run / 'static_field_strength.npz') == complete['static_sha256']
    cohort = pd.read_csv(run / 'inputs.csv')
    x = np.stack([r.image_array(r.ROOT / path, hash_) for path, hash_ in zip(cohort.image_path, cohort.image_sha256)])
    with np.load(run / 'static_field_strength.npz', allow_pickle=False) as pack:
        static = {name: pack[name] for name in pack.files}
    for name, expected in r.static_images(cohort, x).items():
        np.testing.assert_array_equal(static[name], expected)
    result = {'run': run_id, 'protocol_sha256': complete['protocol_sha256'],
              'fresh_real_prediction_rows': real_check(run, protocol, cohort), 'epochs': {}}
    for epoch in protocol['field_strength_epochs']:
        result['epochs'][str(epoch)] = phase_check(run / f'epoch{epoch}_field_strength', protocol, cohort, static,
                                                   'field_strength')
    mfg = run / 'epoch10_manufacturer'
    if (mfg / 'complete.json').is_file():
        result['manufacturer'] = phase_check(mfg, protocol, cohort, {}, 'manufacturer')
    out.mkdir(exist_ok=False)
    r.json_write(out / 'verified.json', result)
    (out / 'README.md').write_text(
        f'Produced by `python -m scripts.fresh_duke.verify_replication --run-id {run_id} '
        f'--verification-id {verification_id}`. verified.json checks complete coverage, hashes, exact '
        'static-control replay, both primary judges\' probabilities, hard decisions and paired bootstrap.\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--verification-id', required=True)
    args = p.parse_args()
    main(args.run_id, args.verification_id)
