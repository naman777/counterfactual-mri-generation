"""Freeze, dry-run and execute the same-patient Duke judge replication locally."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fresh_duke.paired_stats import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, PRIMARY_JUDGES, patient_estimate, primary_pairs, primary_report, summarise,
)
from scripts.v4_convergence.core import AXES, digest, json_write, load_models, probabilities, sample, target
from scripts.v4_heldout.audit import ROOT
from scripts.v4_heldout.download import valid_id
from scripts.v2_quickwins.judge_model import load_judge
from src.inference.baselines import gaussian_noise_baseline, histogram_matching_baseline, identity_baseline
from src.models.predictor import AcquisitionPredictor


REPORT = ROOT / 'reports/v22_duke_replication'
LOCK = ROOT / 'reports/v21_fresh_duke/acquisition_20261118_02'
VERIFIED = ROOT / 'reports/v21_fresh_duke/verification_20261118_02/verified.json'
OLDER = ROOT / 'reports/v2_quickwins/fs_t100_w5/field_strength'
HELDOUT = ROOT / 'reports/v4_heldout_judge/b2_local_20261016_01'
B1 = ROOT / 'reports/v4_diffusion/b1_20261016_01'
TRAIN = ROOT / 'data/processed/ax_t1_128/splits/train.csv'
TEST = TRAIN.with_name('test.csv')
DIFFUSION = {
    10: 'checkpoints/diffusion/ax_t1_128/latest.pt',
    110: 'checkpoints/diffusion/ax_t1_128_v4/b1_20261016_01/selected_epoch_0110.pt',
}
GENERATED = ('target', 'source_label_null', 'unconditional_w0')
STATIC = ('identity', 'gaussian_noise', 'train_histogram')
BATCH = 8
HISTORICAL_PIXEL_TOLERANCE = 0.0005


def save_csv(path, table):
    with path.open('x', encoding='utf-8', newline='') as stream:
        table.to_csv(stream, index=False)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def image_array(path, expected_hash=None):
    if expected_hash is not None and digest(path) != expected_hash:
        raise ValueError(f'Array bytes changed: {path}')
    image = np.load(path, allow_pickle=False)
    if image.dtype != np.float32 or image.shape != (128, 128) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f'Invalid normalised image: {path}')
    return image


def judge_specs():
    original = 'checkpoints/predictor/ax_t1_128/best.pt'
    specs = {PRIMARY_JUDGES[0]: original}
    for family, suffix in [('training_robust', ''), ('training_flip_only', '_plain')]:
        for seed in range(3):
            specs[f'{family}_seed{seed}'] = f'reports/v2_quickwins/robust_judge/judge_resnet18{suffix}_seed{seed}.pt'
    for row in load_json(HELDOUT / 'frozen_checkpoints.json'):
        path = HELDOUT / row['checkpoint']
        if digest(path) != row['sha256']:
            raise ValueError(f'Historical judge checkpoint changed: {path}')
        specs[f"heldout_{row['family']}_seed{row['seed']}"] = rel(path)
    if len(specs) != 13:
        raise ValueError('Unexpected judge roster')
    return specs


def load_assessor(path, name, device):
    if name == PRIMARY_JUDGES[0]:
        model = AcquisitionPredictor().to(device)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        return model.eval()
    return load_judge(path, device)


def classify(path, name, images, device):
    model = load_assessor(path, name, device)
    return probabilities(model, images, original=name == PRIMARY_JUDGES[0], batch=32)


def freeze(run):
    verify = load_json(VERIFIED)
    lock = load_json(LOCK / 'lock.json')
    if verify['status'] != 'target_met' or digest(LOCK / 'lock.json') != verify['lock_sha256']:
        raise ValueError('Duke source lock is absent or changed')
    for name, expected in lock['files'].items():
        if digest(LOCK / name) != expected:
            raise ValueError(f'Changed acquisition manifest: {name}')
    cohort = pd.read_csv(LOCK / 'slice_manifest.csv').sort_values(['patient_id', 'physical_index']).reset_index(drop=True)
    cohort = cohort.rename(columns={'sop_uid': 'slice_id'})
    cohort['mfg'] = cohort.manufacturer.map({'GE': 0, 'Siemens': 1})
    cohort['fs'] = cohort.field_strength.map({'1.5T': 0, '3.0T': 1})
    cohort['seed'] = np.arange(42, 42 + len(cohort))
    if not cohort.slice_id.is_unique or len(cohort) != verify['reproduced_arrays'] or not cohort.groupby('patient_id').size().eq(5).all():
        raise ValueError('Unexpected frozen cohort shape')
    train = pd.read_csv(TRAIN)
    if set(cohort.patient_id) & set(train.patient_id):
        raise ValueError('Fresh patient is present in histogram training bank')
    rng = np.random.default_rng(42)
    spectrum = [train.index[train.field_strength_label.eq(1 - x)].to_numpy() for x in cohort.fs]
    references = train.loc[[rng.choice(indices) for indices in spectrum]].reset_index(drop=True)
    cohort['hist_reference_sop'] = references.sop_uid
    cohort['hist_reference_patient'] = references.patient_id
    cohort['hist_reference_path'] = references.image_path
    cohort['hist_reference_sha256'] = [digest(ROOT / p) for p in references.image_path]
    save_csv(run / 'inputs.csv', cohort)
    judges = judge_specs()
    dependencies = [Path(__file__), Path(__file__).with_name('paired_stats.py'), TRAIN, TEST,
                    LOCK / 'lock.json', LOCK / 'slice_manifest.csv', VERIFIED, OLDER / 'field_strength_results.csv',
                    HELDOUT / 'regrade_old/per_slice.csv', HELDOUT / 'frozen_checkpoints.json',
                    HELDOUT / 'inputs/old_sets.json', B1 / 'inputs/manifest.json']
    dependencies += [ROOT / p for p in [
        'scripts/v4_convergence/core.py', 'scripts/v2_quickwins/judge_model.py',
        'src/inference/baselines.py', 'src/models/diffusion_unet.py', 'src/models/conditioning.py',
        'src/models/ddpm_scheduler.py', 'src/models/predictor.py',
        *judges.values(), *DIFFUSION.values(),
    ]]
    hashed = {rel(path): digest(path) for path in dependencies}
    for historical_name, path in [('initial.pt', DIFFUSION[10]), ('predictor.pt', judges[PRIMARY_JUDGES[0]])]:
        ref = load_json(B1 / 'inputs/manifest.json')['assets'][historical_name]['sha256']
        if hashed[path] != ref:
            raise ValueError(f'Changed historical asset: {path}')
    freeze_record = {
        'input_sha256': digest(run / 'inputs.csv'), 'files_sha256': hashed,
        'checkpoints': DIFFUSION, 'judges': judges, 'primary_judges': PRIMARY_JUDGES,
        'primary': {'epoch': 10, 'axis': 'field_strength', 't': 100, 'w': 5,
                    'estimate': 'mean across patients of within-patient mean paired hit differences',
                    'direction': 'flip source field strength; preserve manufacturer and source EchoTime',
                    'sensitivity': 'both primary judges correct on the same native source slice'},
        'field_strength_epochs': [10, 110], 'real_transforms': ['native', 'vertical_flip'],
        'generated_controls': GENERATED, 'static_controls': STATIC, 'gaussian_sigma': .03,
        'histogram_design': 'one original training slice of each respective target class, fixed SHA256; numpy RNG 42',
        'control_hit_definition': 'count each control against the requested opposite label; retain source agreement too',
        'seed': '42 + zero-based position ordered by patient ID then physical slice index; shared for all sampler arms and epochs',
        'resampling': {'seed': BOOTSTRAP_SEED, 'draws': BOOTSTRAP_DRAWS,
                       'historical': 'patient percentile bootstrap without strata',
                       'fresh': 'resample patients separately within fixed manufacturer × source field-strength strata'},
        'sampler': {'t': 100, 'cfg': 5, 'batch': BATCH, 'torch': torch.__version__,
                    'device': torch.cuda.get_device_name(), 'cudnn_tf32': True, 'matmul_tf32': False},
        'historical_smoke': {'archive_decisions': 'all 100 paired decisions from both primary judges must agree',
                             'generation_examples': 'first archived slice in each direction',
                             'max_pixel_tolerance': HISTORICAL_PIXEL_TOLERANCE,
                             'generated_decisions': 'both primary judges must keep the same hard decisions'},
        'minimum': {'patients': 44, 'per_cell': 11},
        'retry': 'one unchanged retry for a failed generation batch; retain each failed arm/range and exception',
        'primary_completeness': 'Report observed paired slices; full cohort required for a complete confirmatory run',
        'optional_manufacturer': {'epoch': 10, 't': 300, 'w': 5, 'arms': ['target', 'source_label_null'],
                                  'after_field_strength': True},
    }
    json_write(run / 'protocol.json', freeze_record)
    print(f'FROZEN {len(cohort)} source slices and {len(judges)} judges', flush=True)
    return freeze_record


def read_freeze(run):
    protocol = load_json(run / 'protocol.json')
    for path, expected in protocol['files_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    if digest(run / 'inputs.csv') != protocol['input_sha256'] or torch.__version__ != protocol['sampler']['torch']:
        raise ValueError('Frozen source index or Torch version changed')
    return protocol


def historical(run, protocol):
    original = pd.read_csv(OLDER / 'field_strength_results.csv')
    held = pd.read_csv(HELDOUT / 'regrade_old/per_slice.csv').query(
        "set == 'old_fs_t100_w5' and variant == 'cf' and family == 'orientation_robust' and seed == 0"
    )
    historical_set = next(x for x in load_json(HELDOUT / 'inputs/old_sets.json') if x['name'] == 'old_fs_t100_w5')
    test = pd.read_csv(TEST).iloc[historical_set['test_indices']].reset_index(drop=True)
    rows = original.merge(held[['patient_id', 'example_id', 'test_index', 'hit']],
                          on=['patient_id', 'example_id'], validate='one_to_one')
    assert len(rows) == len(test) == len(original) == len(held) == 100
    assert test.patient_id.equals(rows.patient_id) and rows.test_index.tolist() == historical_set['test_indices']
    assert [Path(p) for p in test.image_path] == [Path(p) for p in rows.image_path]
    files = [OLDER / 'counterfactuals' / f'{identifier}_counterfactual.npy' for identifier in rows.example_id]
    edited = np.stack([image_array(p, historical_set['provenance']['image_sha256'][p.name]) for p in files])
    native = np.stack([image_array(ROOT / p) for p in test.image_path])
    context = pd.DataFrame({'slice_id': rows.example_id, 'patient_id': rows.patient_id,
                            'mfg': test.manufacturer_label, 'fs': test.field_strength_label})
    context['target_label'] = 1 - context.fs
    results = []
    for name, indicator in zip(PRIMARY_JUDGES, ('fs_hit', 'hit'), strict=True):
        path = ROOT / protocol['judges'][name]
        model = load_assessor(path, name, 'cuda')
        before = probabilities(model, native, original=name == PRIMARY_JUDGES[0], batch=32)['fs']
        after = probabilities(model, edited, original=name == PRIMARY_JUDGES[0], batch=32)['fs']
        hit = after.argmax(1) == context.target_label.to_numpy()
        np.testing.assert_array_equal(hit, rows[indicator].astype(bool))
        results.append(context.assign(judge=name, hit=hit, native_source_correct=before.argmax(1) == context.fs.to_numpy()))
    pair = primary_pairs(pd.concat(results, ignore_index=True))
    summary, patient = primary_report(pair, stratified=False)
    if summary['all']['agreement_counts'] != {'both_hit': 9, 'original_only': 85, 'heldout_only': 0, 'neither': 6}:
        raise ValueError('Historical agreement table differs from submitted-plan record')
    out = run / 'historical'
    out.mkdir(exist_ok=False)
    save_csv(out / 'paired_slices.csv', pair)
    save_csv(out / 'patient_differences.csv', patient)
    json_write(out / 'primary.json', summary)
    print('HISTORICAL GATE PASSED', json.dumps(summary['all']), flush=True)
    return context, native, edited


def conds(cohort):
    return {'mfg': cohort.mfg.to_numpy(), 'fs': cohort.fs.to_numpy(), 'echo': cohort.echo_time_norm.to_numpy()}


def discard_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def sample_routes(models, x, source, seeds, axis, timestep, arms):
    requested = target(source, axis)
    for arm in arms:
        yield arm, sample(*models, requested if arm == 'target' else source, seeds,
                          timestep, 0 if arm == 'unconditional_w0' else 5, x)


def static_images(cohort, x):
    refs = [image_array(ROOT / p, digest_) for p, digest_ in zip(cohort.hist_reference_path, cohort.hist_reference_sha256)]
    return {
        'identity': np.stack([identity_baseline(image) for image in x]),
        'gaussian_noise': np.stack([gaussian_noise_baseline(image, .03, int(seed)) for image, seed in zip(x, cohort.seed)]),
        'train_histogram': np.stack([histogram_matching_baseline(image, ref) for image, ref in zip(x, refs)]),
    }


def historical_smoke(run, protocol, context, native, edited):
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    locations = next(x['test_indices'] for x in load_json(HELDOUT / 'inputs/old_sets.json') if x['name'] == 'old_fs_t100_w5')
    pick = [int(np.flatnonzero(context.fs.eq(i))[0]) for i in range(2)]
    demo = context.iloc[pick].copy()
    demo['echo_time_norm'] = test.iloc[np.asarray(locations)[pick]].echo_time_norm.to_numpy()
    demo['seed'] = np.asarray(pick) + 42
    hist_rng = np.random.default_rng(42)
    indices = [hist_rng.choice(train.index[train.field_strength_label.eq(1 - value)]) for value in demo.fs]
    refs = train.loc[indices]
    demo['hist_reference_path'] = refs.image_path.to_numpy()
    demo['hist_reference_sha256'] = [digest(ROOT / path) for path in refs.image_path]
    x = native[pick]
    static = static_images(demo, x)
    np.testing.assert_array_equal(static['identity'], x)
    (run / 'historical/pipeline').mkdir()
    demo_dir = run / 'historical/pipeline'
    native_grades = real_accuracy(demo_dir, demo, x, protocol)
    results = {}
    for epoch in protocol['field_strength_epochs']:
        c, u, s, ck = load_models(ROOT / protocol['checkpoints'][str(epoch)], 'cuda')
        if ck['epoch'] != epoch:
            raise ValueError('Unexpected epoch metadata')
        generated = dict(sample_routes((c, u, s), x, conds(demo), demo.seed, 'field_strength', 100, GENERATED))
        results[str(epoch)] = {name: float(np.abs(images - x).mean()) for name, images in {**static, **generated}.items()}
        if epoch == 10:
            delta = np.abs(edited[pick] - generated['target'])
            results['historical_regeneration_max_abs'] = float(delta.max())
            if delta.max() > HISTORICAL_PIXEL_TOLERANCE:
                raise ValueError(f'Historical sampler drift on the frozen direction examples: {delta.max()}')
            for judge in PRIMARY_JUDGES:
                model = load_assessor(ROOT / protocol['judges'][judge], judge, 'cuda')
                replay = probabilities(model, np.concatenate([edited[pick], generated['target']]),
                                       original=judge == PRIMARY_JUDGES[0], batch=32)['fs'].argmax(1)
                np.testing.assert_array_equal(replay[:len(pick)], replay[len(pick):])
                del model
        del c, u, s, ck
        discard_cuda()
        demo_stage = demo_dir / f'epoch{epoch}_field_strength'
        (demo_stage / 'chunks').mkdir(parents=True)
        with (demo_stage / 'chunks/0000.npz').open('xb') as stream:
            np.savez(stream, **generated, offsets=np.arange(len(pick)))
        n, paired = grade(demo_stage, demo, x, protocol, native_grades, 'field_strength', epoch, static, announce=False)
        if n != len(pick) * len(protocol['judges']) * (len(STATIC) + len(GENERATED)) or paired != len(pick):
            raise ValueError('Historical smoke grade coverage differs from expected')
    json_write(run / 'historical/smoke.json', {'tested_historical_examples': demo.slice_id.tolist(), 'results': results})
    print('HISTORICAL GENERATION AND CONTROLS PASSED', flush=True)


def record_base(cohort):
    return cohort[['slice_id', 'patient_id', 'mfg', 'fs', 'seed']].reset_index(drop=True)


def real_accuracy(run, cohort, x, protocol):
    out = run / 'real'
    out.mkdir(exist_ok=False)
    frames = []
    for name, path in protocol['judges'].items():
        model = load_assessor(ROOT / path, name, 'cuda')
        for orientation, images in [('native', x), ('vertical_flip', np.flip(x, axis=1))]:
            pred = probabilities(model, images, original=name == PRIMARY_JUDGES[0], batch=32)
            for axis, label in AXES.items():
                truth = cohort[label].to_numpy()
                frames.append(record_base(cohort).assign(judge=name, orientation=orientation, axis=axis,
                    truth=truth, p0=pred[label][:, 0], p1=pred[label][:, 1],
                    prediction=pred[label].argmax(1), correct=pred[label].argmax(1) == truth))
        del model
    grades = pd.concat(frames, ignore_index=True)
    save_csv(out / 'per_slice.csv', grades)
    save_csv(out / 'accuracy.csv', summarise(grades, 'correct', ['judge', 'orientation', 'axis']))
    native = grades.loc[grades.orientation.eq('native')].set_index(['judge', 'axis', 'slice_id'])
    flipped = grades.loc[grades.orientation.eq('vertical_flip')].set_index(['judge', 'axis', 'slice_id'])
    comparison = native.reset_index()
    comparison['flip_minus_native_accuracy'] = flipped.loc[native.index, 'correct'].to_numpy(dtype=int) - native.correct.to_numpy(dtype=int)
    flip_summary = [
        {'judge': j, 'axis': a, **patient_estimate(f, 'flip_minus_native_accuracy', True)}
        for (j, a), f in comparison.groupby(['judge', 'axis'])
    ]
    save_csv(out / 'flip_contrast.csv', pd.DataFrame(flip_summary))
    json_write(out / 'complete.json', {'prediction_rows': len(grades), 'per_slice_sha256': digest(out / 'per_slice.csv')})
    print('REAL NATIVE AND VERTICAL FLIP COMPLETE', flush=True)
    return native


def generate(out, cohort, x, protocol, epoch, axis, timestep, arms):
    c, u, sched, checkpoint = load_models(ROOT / protocol['checkpoints'][str(epoch)], 'cuda')
    if checkpoint['epoch'] != epoch:
        raise ValueError('Frozen checkpoint epoch mismatch')
    del checkpoint
    start, failure_count = time.perf_counter(), 0
    chunks = out / 'chunks'
    chunks.mkdir()
    with (out / 'generation.jsonl').open('x', encoding='utf-8') as events:
        for offset in range(0, len(x), BATCH):
            stop = min(len(x), offset + BATCH)
            group = cohort.iloc[offset:stop]
            pack = {}
            for arm in arms:
                for attempt in (1, 2):
                    try:
                        pack[arm] = dict(sample_routes((c, u, sched), x[offset:stop], conds(group), group.seed,
                                                       axis, timestep, [arm]))[arm]
                    except (RuntimeError, FloatingPointError) as exc:
                        print(json.dumps({'offset': offset, 'stop': stop, 'arm': arm, 'attempt': attempt, 'status': 'failed',
                                          'slices': group.slice_id.tolist(), 'error': repr(exc)}), file=events, flush=True)
                        if attempt == 2:
                            failure_count += stop - offset
                    else:
                        break
                    discard_cuda()
            path = chunks / f'{offset:04d}.npz'
            with path.open('xb') as stream:
                np.savez(stream, **pack, offsets=np.arange(offset, stop))
            print(json.dumps({'offset': offset, 'stop': stop, 'status': 'saved', 'arms': list(pack),
                              'sha256': digest(path)}), file=events, flush=True)
            if offset % (BATCH * 4) == 0 or stop == len(x):
                print(f'EPOCH {epoch} {axis}: {stop}/{len(x)} source slices, {time.perf_counter()-start:.0f}s', flush=True)
    del c, u, sched
    discard_cuda()
    return failure_count


def grade(out, cohort, x, protocol, native, axis, epoch, static, announce=True):
    variants = {}
    for path in sorted((out / 'chunks').glob('*.npz')):
        with np.load(path, allow_pickle=False) as chunk:
            for key in set(chunk.files) - {'offsets'}:
                variants.setdefault(key, []).append((chunk['offsets'], chunk[key]))
    for key, images in static.items():
        variants[key] = [(np.arange(len(x)), images)]
    arrays = {key: (np.concatenate([v[0] for v in chunks]), np.concatenate([v[1] for v in chunks]))
              for key, chunks in variants.items()}
    frames = []
    edited = AXES[axis]
    names = {v: k for k, v in AXES.items()}
    for judge, path in protocol['judges'].items():
        model = load_assessor(ROOT / path, judge, 'cuda')
        for arm in (*GENERATED, *STATIC):
            if arm not in arrays:
                continue
            offsets, images = arrays[arm]
            group = cohort.iloc[offsets]
            pred = probabilities(model, images, original=judge == PRIMARY_JUDGES[0], batch=32)
            after = pred[edited].argmax(1)
            source = group[edited].to_numpy()
            target_labels = 1 - source
            before = native.loc[(judge, axis)].loc[group.slice_id]
            row = record_base(group).assign(
                epoch=epoch, axis=axis, judge=judge, arm=arm, source_label=source, target_label=target_labels,
                p0=pred[edited][:, 0], p1=pred[edited][:, 1], prediction=after, hit=after == target_labels,
                source_agreement=after == source, native_source_correct=before.correct.to_numpy(),
                target_probability_gain=pred[edited][np.arange(len(after)), target_labels] -
                before[['p0', 'p1']].to_numpy()[np.arange(len(after)), target_labels],
                abs_image_change=np.abs(images - x[offsets]).mean(axis=(1, 2)),
            )
            other = 'mfg' if edited == 'fs' else 'fs'
            other_before = native.loc[(judge, names[other])].loc[group.slice_id]
            row = row.assign(other_p0=pred[other][:, 0], other_p1=pred[other][:, 1],
                             other_stability=pred[other].argmax(1) == other_before.prediction.to_numpy())
            frames.append(row)
        del model
    grades = pd.concat(frames, ignore_index=True)
    save_csv(out / 'per_slice.csv', grades)
    keys = ['epoch', 'axis', 'judge', 'arm']
    save_csv(out / 'hits.csv', summarise(grades, 'hit', keys))
    paired_controls = []
    target_rows = grades.query("arm == 'target'")
    for arm, group in grades.query("arm != 'target'").groupby('arm'):
        columns = ['slice_id', 'patient_id', 'mfg', 'fs', 'judge', 'target_label', 'hit']
        paired = target_rows[columns].merge(group[columns], on=columns[:-1], suffixes=('_target', '_control'),
                                            validate='one_to_one')
        paired['difference'] = paired.hit_target.astype(int) - paired.hit_control.astype(int)
        paired_controls.append(paired.assign(control=arm))
    if paired_controls:
        controls = pd.concat(paired_controls, ignore_index=True)
        save_csv(out / 'paired_controls.csv', controls)
        save_csv(out / 'control_differences.csv', summarise(controls, 'difference', ['judge', 'control']))
    pair = primary_pairs(target_rows)
    if axis == 'field_strength':
        summary, patients = primary_report(pair, stratified=True)
        save_csv(out / 'paired_judges.csv', pair)
        save_csv(out / 'patient_differences.csv', patients)
        json_write(out / 'primary.json', summary)
        if announce:
            print(f'EPOCH {epoch} FIELD STRENGTH PAIRED', json.dumps(summary['all']), flush=True)
    return len(grades), len(pair)


def execute(run, protocol, optional_manufacturer):
    if not (run / 'historical/smoke.json').exists():
        raise ValueError('The historical gate must pass before any fresh image is loaded')
    cohort = pd.read_csv(run / 'inputs.csv')
    x = np.stack([image_array(ROOT / p, sha) for p, sha in zip(cohort.image_path, cohort.image_sha256)])
    native = real_accuracy(run, cohort, x, protocol)
    static = static_images(cohort, x)
    with (run / 'static_field_strength.npz').open('xb') as stream:
        np.savez(stream, **static)
    planned = [(epoch, 'field_strength', 100, GENERATED) for epoch in protocol['field_strength_epochs']]
    if optional_manufacturer:
        planned.append((10, 'manufacturer', 300, ('target', 'source_label_null')))
    results = {}
    for epoch, axis, timestep, arms in planned:
        name = f'epoch{epoch}_{axis}'
        results[name] = run_phase(run / name, cohort, x, protocol, native, epoch, axis, timestep, arms,
                                  static if axis == 'field_strength' else {})
    a, b = [pd.read_csv(run / f'epoch{e}_field_strength/paired_judges.csv') for e in protocol['field_strength_epochs']]
    columns = ['slice_id', 'patient_id', 'mfg', 'fs', 'difference', 'hit_original', 'hit_heldout']
    paired = a[columns].merge(b[columns], on=columns[:4], suffixes=('_e10', '_e110'), validate='one_to_one')
    for metric in columns[4:]:
        paired[metric + '_change'] = paired[metric + '_e110'].astype(int) - paired[metric + '_e10'].astype(int)
    save_csv(run / 'epoch_comparison_slices.csv', paired)
    json_write(run / 'epoch_comparison.json', {m: patient_estimate(paired, m + '_change', True) for m in columns[4:]})
    json_write(run / 'complete.json', {'phases': results, 'protocol_sha256': digest(run / 'protocol.json'),
                                     'static_sha256': digest(run / 'static_field_strength.npz')})


def run_phase(out, cohort, x, protocol, native, epoch, axis, timestep, arms, static):
    out.mkdir(exist_ok=False)
    try:
        failures = generate(out, cohort, x, protocol, epoch, axis, timestep, arms)
        predictions, pairs = grade(out, cohort, x, protocol, native, axis, epoch, static)
    except Exception as exc:
        json_write(out / 'failed.json', {'exception': repr(exc), 'stage': out.name})
        raise
    result = {'predictions': predictions, 'paired_target_slices': pairs, 'failed_arm_slices': failures,
              'status': 'complete' if not failures and pairs == len(cohort) else 'incomplete',
              'generation_log_sha256': digest(out / 'generation.jsonl'),
              'per_slice_sha256': digest(out / 'per_slice.csv')}
    json_write(out / 'complete.json', result)
    return result


def manufacturer_followup(run, protocol):
    complete = load_json(run / 'complete.json')
    if any(complete['phases'][f'epoch{epoch}_field_strength']['status'] != 'complete'
           for epoch in protocol['field_strength_epochs']):
        raise ValueError('Complete both frozen field-strength phases first')
    real = run / 'real/per_slice.csv'
    if digest(real) != load_json(run / 'real/complete.json')['per_slice_sha256']:
        raise ValueError('Fresh real grading changed')
    cohort = pd.read_csv(run / 'inputs.csv')
    x = np.stack([image_array(ROOT / p, sha) for p, sha in zip(cohort.image_path, cohort.image_sha256)])
    native = pd.read_csv(real).query("orientation == 'native'").set_index(['judge', 'axis', 'slice_id'])
    spec = protocol['optional_manufacturer']
    run_phase(run / 'epoch10_manufacturer', cohort, x, protocol, native, spec['epoch'],
              'manufacturer', spec['t'], spec['arms'], {})


def main(run_id, stage, optional_manufacturer):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    run = REPORT / valid_id(run_id)
    if stage in {'freeze', 'all'}:
        run.mkdir(parents=True, exist_ok=False)
        (run / 'README.md').write_text(
            'Produced by `python -m scripts.fresh_duke.replication`. protocol.json and inputs.csv freeze source and reference '
            'hashes, settings, seed streams, checkpoints, assessors and analysis. historical/ contains the paired replay and '
            'control smoke gate. real/ contains native/flip predictions and patient intervals. Epoch folders contain '
            'hashed image chunks, per-slice predictions, matched controls, patient estimates and explicit failures. '
            'The static field-strength controls are shared across epochs. complete.json records terminal coverage.\n',
            encoding='utf-8',
        )
        freeze(run)
    protocol = read_freeze(run)
    if stage in {'historical', 'all'}:
        historical_smoke(run, protocol, *historical(run, protocol))
    if stage in {'fresh', 'all'}:
        execute(run, protocol, optional_manufacturer)
    if stage == 'manufacturer':
        manufacturer_followup(run, protocol)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--stage', choices=['freeze', 'historical', 'fresh', 'all', 'manufacturer'], required=True)
    p.add_argument('--optional-manufacturer', action='store_true')
    args = p.parse_args()
    main(args.run_id, args.stage, args.optional_manufacturer)
