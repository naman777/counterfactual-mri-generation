"""Validation-only target/null grading in source-native coordinates with CIs."""
import argparse
import gc
import json
import re
import time

import numpy as np
import pandas as pd
import torch

from scripts.v7_geometry.orientation import invert_orientation
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_protocol import bootstrap_patients, select_setting, settings
from scripts.v8_generator_dev.full_train import deterministic_runtime, load_ema
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring


def source_native(images, plan):
    images = np.asarray(images)
    return np.stack([invert_orientation(image, plan) for image in images])


@torch.inference_mode()
def predict_judge(model, images, device):
    tensor = torch.from_numpy(np.asarray(images, np.float32)).unsqueeze(1).to(device)
    return {key: value.softmax(1).cpu().numpy() for key, value in model(tensor).items()}


def summarise(rows, config):
    result = []
    keys = ['axis', 'timestep', 'guidance', 'judge', 'variant', 'evaluation_phase']
    for values, group in rows.groupby(keys):
        strata = [('all', group)]
        strata += [(f'direction:{direction}', subset) for direction, subset in group.groupby('direction')]
        strata += [(f'support:{status}', subset) for status, subset in group.groupby('condition_support_status')]
        for stratum, subset in strata:
            for metric in ('target_hit', 'source_correct', 'target_probability', 'other_axis_stable',
                           'ssim', 'mad', 'paired_hit_delta'):
                result.append(dict(zip(keys, values), stratum=stratum, metric=metric,
                                   generation_seeds=int(subset.generation_seed.nunique()),
                                   **bootstrap_patients(subset, metric, config['evaluation']['bootstrap_seed'],
                                                        config['evaluation']['bootstrap_resamples']),
                                   anatomy_validated=False,
                                   interval='Equal-patient bootstrap, averaging generation seeds within patient'))
    return pd.DataFrame(result)


def evaluate(prepared, checkpoint, run_id, device='cuda', engineering=False):
    from scripts.v2_quickwins.judge_model import load_judge
    from scripts.v5_anatomy.metrics import diagnostics
    started = time.perf_counter()

    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid evaluation run ID')
    prepared, checkpoint = project_path(prepared), project_path(checkpoint)
    protocol = json.loads((prepared / 'protocol.json').read_text())
    config = protocol['config']
    inputs = json.loads((prepared / 'inputs.json').read_text())
    verify_inputs(inputs)
    inputs[checkpoint.relative_to(ROOT).as_posix()] = digest(checkpoint)
    inputs[(prepared / 'protocol.json').relative_to(ROOT).as_posix()] = digest(prepared / 'protocol.json')
    deterministic_runtime()
    c, model, scheduler, ck = load_ema(checkpoint, device)
    if ck['config'] != config or ck['protocol_sha256'] != digest(prepared / 'protocol.json'):
        raise ValueError('Checkpoint belongs to another frozen protocol')
    if ck['engineering'] and not engineering:
        raise ValueError('Engineering checkpoints cannot be graded as full-training efficacy')
    train, val = [CanonicalDataset(config['view'], split) for split in ('train', 'val')]
    selected_path = prepared / 'validation_selection.csv'
    indices = pd.read_csv(selected_path).dataset_index.astype(int).tolist()
    inputs[selected_path.relative_to(ROOT).as_posix()] = digest(selected_path)
    if engineering:
        indices = indices[:2]
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new validation run ID')
    output.mkdir(parents=True)
    (output / 'arrays').mkdir()
    json_write(output / 'protocol.json', dict(config=config, checkpoint=str(checkpoint.relative_to(ROOT)),
                                              checkpoint_sha256=digest(checkpoint), engineering=engineering,
                                              effective_settings='t=2,w=3,first two preselected patients' if engineering else 'frozen selection grid and confirmation seeds',
                                              evaluation_frame='canonical arrays; inverse-source transform for frozen judges',
                                              scope='Development validation only; anatomy/pathology outcomes unknown'))
    (output / 'README.md').write_text(
        '# Canonical generator validation\n\nProducer: `python -m scripts.v8_generator_dev.evaluate_full`.\n'
        'protocol.json fixes checkpoint, frames, patients and evaluation mode. '
        'arrays/case_*.npz retains every source/null/target canonical image and conditions. '
        'requests.csv maps cases to source, target, fixed EchoTime, settings and support. '
        'per_image.csv contains every control, named judge, probability/hit, paired delta '
        'and correspondence/proxy metric; missing anatomy validation remains unknown. '
        'summary.csv gives patient-bootstrap intervals by direction/support and overall. '
        'selection.json records all provisional candidates or an explicit infeasible result; '
        'engineering runs select none. inputs.json hashes sources/code/judges/checkpoint; '
        'complete.json records scope and actual request counts. No test pixels or judge gradients.\n', encoding='utf-8')
    judges = {}
    for seed, path in enumerate(config['evaluation']['judge_paths']):
        path = project_path(path)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        judges[f'heldout_orientation_robust_seed{seed}'] = load_judge(path, device).eval().requires_grad_(False)
    records, requests = [], []
    evaluation = config['evaluation']
    sources = {}
    for index in indices:
        item = val[index]
        canonical = val.image_01(index)
        native = source_native(np.stack([canonical]), val.plans[index])
        sources[index] = (item, canonical, {name: predict_judge(judge, native, device) for name, judge in judges.items()})

    def run_cases(axis, points, seeds, phase):
        for point in points:
            for generation_seed in seeds:
                for index in indices:
                    if time.perf_counter() - started >= config['resources'].get('max_validation_wall_seconds', 1800):
                        csv_write(output / 'partial_requests.csv', pd.DataFrame(requests))
                        csv_write(output / 'partial_per_image.csv', pd.DataFrame(records))
                        json_write(output / 'resource_stop.json', dict(complete=False, reason='validation_wall_time_ceiling',
                                                                       retained_cases=len(requests), anatomy_validated=False))
                        raise TimeoutError('Validation resource cap; partial outputs retained, no completed selection')
                    item, source, source_probs = sources[index]
                    target_m, target_f = item['manufacturer_label'], item['field_strength_label']
                    original = target_m if axis == 'manufacturer' else target_f
                    target_m, target_f = (1 - target_m, target_f) if axis == 'manufacturer' else (target_m, 1 - target_f)
                    target_label = 1 - original
                    support = request_support(train.frame, target_m, target_f, float(item['echo_time_norm']),
                                              config['data']['minimum_cell_patients'])
                    image = item['image'].unsqueeze(0).to(device)
                    echo = torch.tensor([item['echo_time_norm']], device=device)
                    seed = int(generation_seed) + int(index)
                    arrays = {'identity': source}
                    for variant, mfg, fs in [('null', item['manufacturer_label'], item['field_strength_label']),
                                             ('target', target_m, target_f)]:
                        arrays[variant] = sample_wiring(c, model, scheduler, image,
                            torch.tensor([mfg], device=device), torch.tensor([fs], device=device), echo,
                            seed, point['timestep'], point['guidance'])[0, 0]
                    case = len(requests)
                    path = output / 'arrays' / f'case_{case:05d}.npz'
                    with path.open('xb') as handle:
                        np.savez_compressed(handle, source=source, null=arrays['null'], target=arrays['target'],
                                            source_echo_time_norm=item['echo_time_norm'],
                                            target_echo_time_norm=item['echo_time_norm'],
                                            target_manufacturer=target_m, target_field_strength=target_f)
                    request = dict(case=case, patient_id=item['patient_id'], dataset_index=index, axis=axis,
                                   direction=f'{original}_to_{target_label}', generation_seed=int(generation_seed),
                                   actual_sampling_seed=seed, evaluation_phase=phase,
                                   timestep=point['timestep'], guidance=point['guidance'],
                                   source_echo_time_norm=float(item['echo_time_norm']),
                                   target_echo_time_norm=float(item['echo_time_norm']),
                                   target_manufacturer=target_m, target_field_strength=target_f,
                                   array_file=str(path.relative_to(output)), **support)
                    requests.append(request)
                    native = source_native(np.stack([arrays[name] for name in ('identity', 'null', 'target')]), val.plans[index])
                    metrics = {name: diagnostics(source, arrays[name]) for name in arrays}
                    key, other_key = ('mfg', 'fs') if axis == 'manufacturer' else ('fs', 'mfg')
                    for name, judge in judges.items():
                        probabilities = predict_judge(judge, native, device)
                        before_other = int(source_probs[name][other_key][0].argmax())
                        null_hit = int(probabilities[key][1].argmax() == target_label)
                        for j, variant in enumerate(('identity', 'null', 'target')):
                            hit = int(probabilities[key][j].argmax() == target_label)
                            records.append(dict(request, variant=variant, judge=name,
                                target_hit=hit, target_probability=float(probabilities[key][j, target_label]),
                                source_correct=int(source_probs[name][key][0].argmax() == original),
                                other_axis_stable=int(probabilities[other_key][j].argmax() == before_other),
                                paired_hit_delta=hit - null_hit,
                                judge_input_frame='source_native_inverse_geometry',
                                anatomy_assessment='unknown_no_independent_review', **metrics[variant]))
                    if case % 10 == 0:
                        print('VALIDATED CASE', case, axis, phase, flush=True)
    if engineering:
        for axis in ('manufacturer', 'field_strength'):
            run_cases(axis, [dict(timestep=2, guidance=3.0)], [evaluation['selection_generation_seed']], 'engineering')
        selection = {axis: dict(selected=None, reason='engineering_check_does_not_select_settings')
                     for axis in ('manufacturer', 'field_strength')}
    else:
        for axis in ('manufacturer', 'field_strength'):
            run_cases(axis, settings(config, axis), [evaluation['selection_generation_seed']], 'selection')
        table = pd.DataFrame(records)
        selection = {axis: select_setting(table, axis, config) for axis in ('manufacturer', 'field_strength')}
        for axis in ('manufacturer', 'field_strength'):
            points = [evaluation['fixed_settings'][axis]]
            chosen = selection[axis]['selected']
            if chosen is not None and chosen not in points:
                points.append(chosen)
            run_cases(axis, points, evaluation['confirmation_generation_seeds'], 'confirmation_same_patients')
    table = pd.DataFrame(records)
    csv_write(output / 'requests.csv', pd.DataFrame(requests))
    csv_write(output / 'per_image.csv', table)
    csv_write(output / 'summary.csv', summarise(table, config))
    json_write(output / 'selection.json', selection)
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(
        complete=True, engineering=engineering, cases=len(requests), patients=len(indices),
        judges=list(judges), both_axes=True, source_echo_time_fixed=True,
        target_null_noise_matched=True, canonical_arrays_scored_in_source_native_frame=True,
        test_pixels_used=False, judge_gradients_used=False, anatomy_validated=False,
        efficacy_established=False, clinical_preservation='unknown',
    ))
    del c, model, scheduler, judges, ck
    gc.collect()
    print('VALIDATION COMPLETE', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    evaluate(args.prepared, args.checkpoint, args.run_id, args.device, args.engineering)
