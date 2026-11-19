"""Run a validation-only paired pilot with frozen models and append-only outputs."""
import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.v4_convergence.core import digest, json_write, load_models, probabilities, target, sample
from scripts.v2_quickwins.judge_model import load_judge
from scripts.v5_anatomy.anchored_sampler import sample_anchored
from scripts.v5_anatomy.metrics import diagnostics

ROOT = Path(__file__).resolve().parents[2]
B2 = ROOT/'reports/v4_heldout_judge/b2_local_20261016_01'


def project_path(value):
    path = Path(str(value).replace('\\', '/'))
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Input must stay inside the project')
    return path


def read_splits():
    splits = {s: pd.read_csv(ROOT/f'data/processed/ax_t1_128/splits/{s}.csv') for s in ['train','val','test']}
    patients = {s: set(d.patient_id.astype(str)) for s,d in splits.items()}
    if any(patients[a]&patients[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Patient split overlap')
    return splits


def select_validation(frame, patients, seed):
    if patients < 1 or patients > frame.patient_id.nunique():
        raise ValueError('Requested patient count is outside validation cohort')
    shuffled = frame.sample(frac=1, random_state=seed)
    return shuffled.drop_duplicates('patient_id').iloc[:patients].copy()


def support_flags(train, src, tgt):
    cell = train[(train.manufacturer_label == int(tgt['mfg'][0])) &
                 (train.field_strength_label == int(tgt['fs'][0]))]
    echo = float(src['echo'][0])
    values = cell.echo_time_norm.to_numpy(float)
    return dict(target_cell_train_patients=int(cell.patient_id.nunique()),
                target_echo_exact_seen=bool(len(values) and np.min(np.abs(values-echo)) <= 1e-5),
                target_echo_in_train_range=bool(len(values) and values.min() <= echo <= values.max()),
                nearest_target_echo_norm_distance=float(np.min(np.abs(values-echo))) if len(values) else None,
                conditioning_warning='protocol-confounded labels; exact EchoTime support is a diagnostic, not proof of causal support')


def bootstrap_mean(values, rng, n):
    values = np.asarray(values, float)
    if not len(values):
        return None, None, None
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarise(frame, protocol):
    rows = []
    metrics = ['target_hit','target_gain','other_axis_stable','proxy_screen_pass','target_hit_and_proxy_screen',
               'ssim','mad','proxy_dice_min','proxy_boundary_hd95_px_max','proxy_centroid_shift_px_max']
    rng = np.random.default_rng(protocol['statistics']['bootstrap_seed'])
    for (axis, variant, judge), group in frame.groupby(['axis','variant','judge'], sort=True):
        # Exactly one slice per patient, so each resampling unit is a patient.
        if group.patient_id.duplicated().any():
            raise ValueError('Summary assumes one slice per patient per setting/judge')
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors='coerce').dropna().to_numpy(float)
            mean, lo, hi = bootstrap_mean(values, rng, protocol['statistics']['bootstrap_resamples'])
            rows.append(dict(axis=axis, variant=variant, judge=judge, metric=metric,
                             patients=len(values), mean=mean, ci95_lo=lo, ci95_hi=hi,
                             interval='patient bootstrap; pilot/degenerate boundaries are not certainty',
                             anatomy_validated=False))
    return pd.DataFrame(rows)


def paired_summary(frame, protocol):
    rng = np.random.default_rng(protocol['statistics']['bootstrap_seed'])
    rows = []
    for (axis, judge), group in frame.groupby(['axis','judge']):
        base = group[group.variant == 'sdedit'].set_index('patient_id')
        for variant in sorted(set(group.variant)-{'sdedit'}):
            candidate = group[group.variant == variant].set_index('patient_id').loc[base.index]
            for metric in ['target_hit','target_hit_and_proxy_screen','proxy_screen_pass','ssim',
                           'proxy_dice_min','proxy_boundary_hd95_px_max']:
                delta = candidate[metric].astype(float)-base[metric].astype(float)
                mean, lo, hi = bootstrap_mean(delta.dropna(), rng, protocol['statistics']['bootstrap_resamples'])
                rows.append(dict(axis=axis, judge=judge, variant=variant, comparator='sdedit',
                                 metric=metric, patients=int(delta.notna().sum()), mean_difference=mean,
                                 ci95_lo=lo, ci95_hi=hi, interpretation='exploratory paired patient difference; no winner selection'))
    return pd.DataFrame(rows)


def png(path, array):
    if path.exists():
        raise FileExistsError(path)
    Image.fromarray(np.round(array*255).astype(np.uint8)).save(path)


def run(config, output, patients=None, device='cuda'):
    config = project_path(config)
    protocol = json.loads(config.read_text(encoding='utf-8'))
    if protocol['split'] != 'val':
        raise ValueError('This development runner permits validation only')
    if patients is not None:
        protocol['patients'] = patients
        protocol['pilot_size_override'] = True
    splits = read_splits()
    selected = select_validation(splits['val'], protocol['patients'], protocol['selection_seed'])
    checkpoint = project_path(protocol['checkpoint'])
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != protocol['expected_checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    frozen_path = B2/'frozen_checkpoints.json'
    frozen = json.loads(frozen_path.read_text())
    entries = [x for x in frozen if x['family'] == 'orientation_robust']
    if sorted(e['seed'] for e in entries) != [0,1,2]:
        raise ValueError('Need all three primary frozen judges')
    for entry in entries:
        if digest(B2/entry['checkpoint']) != entry['sha256']:
            raise ValueError('Frozen judge hash mismatch')
    predictor_path = ROOT/'checkpoints/predictor/ax_t1_128/best.pt'
    inputs = {str(config.relative_to(ROOT)): digest(config), str(checkpoint.relative_to(ROOT)): checkpoint_sha,
              str(frozen_path.relative_to(ROOT)): digest(frozen_path), str(predictor_path.relative_to(ROOT)): digest(predictor_path)}
    for s in splits:
        path = ROOT/f'data/processed/ax_t1_128/splits/{s}.csv'
        inputs[str(path.relative_to(ROOT))] = digest(path)
    sources = []
    for _, row in selected.iterrows():
        path = project_path(row.image_path)
        image = np.load(path).astype(np.float32)
        if image.shape != (128,128) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Invalid source; do not silently change frozen images')
        sources.append(image)
        inputs[str(path.relative_to(ROOT))] = digest(path)
    output = project_path(output)
    if not output.is_relative_to(ROOT/'reports') or output == ROOT/'reports':
        raise ValueError('Outputs must be a new folder under reports/')
    output.mkdir(parents=True, exist_ok=False)
    (output/'arrays').mkdir(); (output/'review').mkdir()
    (output/'README.md').write_text(
        '# Validation anatomy pilot\n\nGenerated by `python -m scripts.v5_anatomy.run_pilot`.\n'
        'This is exploratory validation development, not a test-set result or a clinically validated editor.\n'
        'Thresholded foreground is a contrast-sensitive proxy; human preservation remains pending.\n'
        'No geometric registration, orientation correction, model training, downloads or cloud calls. EchoTime is fixed.\n\n'
        '| Output | Producer and meaning |\n| --- | --- |\n'
        '| protocol.json | run_pilot.py: predeclared settings, engineering gates, cohort and caveats |\n'
        '| inputs.json | run_pilot.py: hashes of frozen inputs and code |\n'
        '| selection.csv | run_pilot.py: one seeded source per validation patient |\n'
        '| arrays/*.npz | run_pilot.py: source, identity, same-noise null, original edit and anchored edits |\n'
        '| generation.csv | run_pilot.py: raw diagnostic metrics and support flags, one row per output |\n'
        '| predictions.csv | run_pilot.py: original and all three held-out judges, gains/hits/stability |\n'
        '| summary.csv, paired_differences.csv | run_pilot.py: patient bootstrap CIs; no winner chosen |\n'
        '| review/ | run_pilot.py: blinded source/output PNGs, mapping, blank review CSV and annotation template |\n'
        '| figures/ | render_pilot.py: contact sheets and paired trade-off charts |\n'
        '| complete.json | run_pilot.py: success receipt, provenance recheck and limitations |\n'
        'Human-annotation scores, if supplied, are created separately by `score_review.py` in a new output folder.\n', encoding='utf-8')
    protocol.update(checkpoint_weights='epoch-110 EMA; unchanged',
                    source_grid='128x128 pixels; no verified physical-distance conversion',
                    judge_role='frozen descriptive development assessors; never gradients or winner selection; no fresh final test claim',
                    runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                 device=device, gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else None),
                    config_source_sha256=digest(config))
    json_write(output/'protocol.json', protocol)
    selected.to_csv(output/'selection.csv', index_label='validation_row_index')
    for path in sorted(Path(__file__).parent.glob('*.py')):
        inputs[str(path.relative_to(ROOT))] = digest(path)
    inputs[str((ROOT/'scripts/v4_convergence/core.py').relative_to(ROOT))] = digest(ROOT/'scripts/v4_convergence/core.py')
    for entry in entries:
        inputs[str((B2/entry['checkpoint']).relative_to(ROOT))] = entry['sha256']
    json_write(output/'inputs.json', inputs)
    torch.set_num_threads(4)
    c, u, sched, ck = load_models(checkpoint, device=device)
    if ck['epoch'] != 110 or 'ema_unet_state_dict' not in ck:
        raise ValueError('Expected epoch-110 EMA checkpoint')
    generation, mappings = [], []
    started = time.perf_counter()
    for axis, setting in protocol['settings'].items():
        for index, ((_, row), image) in enumerate(zip(selected.iterrows(), sources)):
            case_id = f'{axis}_{index:03d}'
            src = dict(mfg=np.array([row.manufacturer_label], np.int64),
                       fs=np.array([row.field_strength_label], np.int64),
                       echo=np.array([row.echo_time_norm], np.float32))
            tgt = target(src, axis)
            assert np.array_equal(tgt['echo'], src['echo'])
            flags = support_flags(splits['train'], src, tgt)
            seed = protocol['sampling_seed']+index
            outputs = {'identity': image.copy()}
            t0 = time.perf_counter()
            for variant, condition, strength in [('sdedit',tgt,0.), ('null',src,0.)] + [
                    (f'anchor_{s:g}',tgt,s) for s in protocol['anchor_strengths']]:
                outputs[variant] = sample_anchored(c,u,sched,condition,[seed],setting['t'],setting['w'],
                                                 image[None], strength, protocol['lowpass_sigma_px'])[0]
                if index == 0 and variant == 'sdedit':
                    reference = sample(c,u,sched,tgt,[seed],setting['t'],setting['w'],image[None])[0]
                    if not np.array_equal(reference, outputs[variant]):
                        raise AssertionError('Zero-strength sampler changed reference output')
            np.savez_compressed(output/'arrays'/f'{case_id}.npz', source=image, **outputs)
            for variant, edited in outputs.items():
                metrics = diagnostics(image, edited, protocol['proxy_thresholds'], protocol['proxy_gates'])
                review_id = hashlib.sha256(f'{protocol["sampling_seed"]}/{case_id}/{variant}'.encode()).hexdigest()[:12]
                png(output/'review'/f'{review_id}_source.png', image)
                png(output/'review'/f'{review_id}_edit.png', edited)
                record = dict(case_id=case_id, review_id=review_id, patient_id=str(row.patient_id),
                              validation_row_index=int(selected.index[index]), axis=axis, variant=variant,
                              seed=seed, timestep=setting['t'], guidance=setting['w'],
                              source_mfg=int(src['mfg'][0]), source_fs=int(src['fs'][0]),
                              target_mfg=int(tgt['mfg'][0]), target_fs=int(tgt['fs'][0]),
                              source_echo=float(src['echo'][0]), target_echo=float(tgt['echo'][0]),
                              source_path=row.image_path, human_preservation_pass=None,
                              deployment_status='research_only_unvalidated', **flags, **metrics)
                generation.append(record)
                mappings.append(dict(review_id=review_id, case_id=case_id, variant=variant,
                                     source_png=f'{review_id}_source.png', edit_png=f'{review_id}_edit.png'))
            print(f'{case_id}: {len(outputs)} outputs, {time.perf_counter()-t0:.1f}s', flush=True)
    del c,u,sched,ck
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    pd.DataFrame(generation).to_csv(output/'generation.csv', index=False)
    mapping = pd.DataFrame(mappings).sample(frac=1, random_state=protocol['selection_seed'])
    mapping.to_csv(output/'review/private_mapping.csv', index=False)
    review = mapping[['review_id','source_png','edit_png']].copy()
    for col in ['reviewer','anatomy_preserved','lesion_assessable','lesion_preserved','comment']:
        review[col] = ''
    review.to_csv(output/'review/review_template.csv', index=False)
    annotation = {'coordinate_order':'x,y in native 128x128 grid; no resizing', 'cases':[
        dict(review_id=m['review_id'], reviewer=None, source_mask=None, edit_mask=None,
             source_landmarks=None, edit_landmarks=None) for m in mapping.to_dict('records')]}
    json_write(output/'review/annotation_template.json', annotation)
    (output/'review/README.md').write_text(
        'Review only the shuffled source/edit PNGs and review_template.csv. Keep private_mapping.csv hidden from readers.\n'
        'These PNGs hide algorithm/checkpoint/judge labels; this is a review preparation pack, not a completed reader study.\n'
        'Leave unassessable fields blank; use yes/no for explicit judgements. Obtain independent qualified review.\n'
        'For annotation_template.json, provide binary .npy source/edit masks in the native grid and named [x,y] landmarks.\n'
        'Masks must be independently checked on each image; do not resize unrelated-sequence labels or infer tumour masks from boxes.\n'
        'The scorer reports partial coverage and never replaces missing human labels with proxy scores.\n', encoding='utf-8')
    from src.models.predictor import AcquisitionPredictor
    predictor = AcquisitionPredictor().to(device)
    predictor.load_state_dict(torch.load(predictor_path, map_location='cpu', weights_only=False)['model_state_dict'])
    models = [('original_predictor', predictor.eval())]
    for entry in entries:
        models.append((f'heldout_orientation_robust{entry["seed"]}', load_judge(B2/entry['checkpoint'], device)))
    rows = []
    for judge, model in models:
        cache = {}
        for record in generation:
            key = record['case_id']
            if key not in cache:
                with np.load(output/'arrays'/f'{key}.npz') as arrays:
                    names = [k for k in arrays.files if k != 'source']
                    probs = probabilities(model, np.stack([arrays['source']] + [arrays[k] for k in names]),
                                          original=judge == 'original_predictor', batch=16)
                cache[key] = names, probs
            names, probs = cache[key]
            i = names.index(record['variant'])+1
            head = 'fs' if record['axis'] == 'field_strength' else 'mfg'
            other = 'mfg' if head == 'fs' else 'fs'
            target_label = record['target_fs' if head == 'fs' else 'target_mfg']
            p0, p1 = probs[head][0], probs[head][i]
            hit = bool(p1.argmax() == target_label)
            rows.append(dict(**record, judge=judge, target_probability_before=float(p0[target_label]),
                             target_probability_after=float(p1[target_label]), target_hit=hit,
                             source_classified_correctly=bool(p0.argmax() == 1-target_label),
                             target_gain=float(p1[target_label]-p0[target_label]),
                             other_axis_stable=bool(probs[other][0].argmax() == probs[other][i].argmax()),
                             target_hit_and_proxy_screen=hit and record['proxy_screen_pass'],
                             target_hit_and_human_preservation=None))
        print('Graded', judge, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'predictions.csv', index=False)
    summarise(frame, protocol).to_csv(output/'summary.csv', index=False)
    paired_summary(frame, protocol).to_csv(output/'paired_differences.csv', index=False)
    for path, expected in inputs.items():
        if digest(ROOT/path) != expected:
            raise AssertionError(f'Input changed during run: {path}')
    json_write(output/'complete.json', dict(complete=True, split='val', patients=len(selected),
               request_pairs=len(selected)*len(protocol['settings']), output_images=len(generation),
               judge_rows=len(frame), seconds=time.perf_counter()-started,
               reference_sampler_bitwise_checked_on_first_case_each_axis=True,
               all_frozen_input_hashes_rechecked=True, anatomy_validated=False,
               human_review_completed=False, model_trained=False, cloud_used=False,
               final_test_evaluation=False, winner_selected=False))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/anatomy_pilot_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--patients', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(run(args.config, args.output, args.patients, args.device), flush=True)
