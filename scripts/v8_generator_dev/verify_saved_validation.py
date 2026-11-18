"""Recompute every saved validation grade without sampling or paid compute."""
import argparse
import gc
import hashlib
import json
import re
import zipfile

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.full_protocol import select_setting, settings
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def same_selection(actual, saved):
    if isinstance(actual, dict):
        return actual.keys() == saved.keys() and all(same_selection(actual[k], saved[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(saved) and all(same_selection(a, b) for a, b in zip(actual, saved))
    if isinstance(actual, float):
        return isinstance(saved, (int, float)) and bool(np.isclose(actual, saved, rtol=0, atol=1e-12))
    return actual == saved


def verify(fetched, training_fetch, bundle, runtime_inputs, run_id, device='cuda'):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new report ID required')
    fetched, training_fetch, bundle = map(project_path, (fetched, training_fetch, bundle))
    runtime_inputs = project_path(runtime_inputs)
    receipt = json.loads((fetched / 'complete.json').read_text())
    if not receipt['validation_retrieved'] or not receipt['checkpoint_hashes_verified']:
        raise ValueError('Completed paired retrieval required')
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Saved validation verification\n\nProducer: `python -m scripts.v8_generator_dev.verify_saved_validation`.\n'
        'protocol.json records scope and cross-runtime float tolerance; input_checks.csv verifies every '
        'frozen input against local originals, the retained approved archive or relocated receipts. '
        'array_checks.csv covers every request/source/condition/SSIM/MAD; grading_checks.csv recomputes '
        'every control with every named frozen judge. inputs.json binds all retrieved files, archive and '
        'verifier. complete.json records verification, never convergence, efficacy or anatomy approval.\n',
        encoding='utf-8')
    deterministic_runtime()
    config = json.loads((training_fetch / 'training/protocol.json').read_text())['config']
    prepared = next((training_fetch / 'protocol').glob('*/protocol.json')).parent
    selected = pd.read_csv(prepared / 'validation_selection.csv').dataset_index.astype(int).tolist()
    val, train = CanonicalDataset(config['view'], 'val'), CanonicalDataset(config['view'], 'train')
    # Float32 probabilities can differ between cloud torch 2.5/A10 and local runtime/GPU.
    # Require identical decisions and paired contrasts; record actual maximum error.
    tolerance = 1e-5
    json_write(output / 'protocol.json', dict(fetched=str(fetched.relative_to(ROOT)),
        training_fetch=str(training_fetch.relative_to(ROOT)), approved_archive=str(bundle.relative_to(ROOT)),
        runtime_inputs=str(runtime_inputs.relative_to(ROOT)),
        device=device, torch_version=torch.__version__, probability_absolute_tolerance=tolerance,
        discrete_grades_must_match_exactly=True, source_and_echo_must_match_exactly=True,
        sampling_repeated=False, test_pixels_used=False))
    inputs = {str(bundle.relative_to(ROOT)): digest(bundle),
              'scripts/v8_generator_dev/verify_saved_validation.py': digest(__file__)}
    for path in fetched.rglob('*'):
        if path.is_file():
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    input_checks, array_checks, grading_checks = [], [], []
    primary_keys = ['dataset_index', 'axis', 'timestep', 'guidance', 'generation_seed']
    common_sources, common_primary = {}, None
    with zipfile.ZipFile(bundle) as archive:
        archived_names = set(archive.namelist())
        for variant in ('additive', 'scale_shift'):
            folder = project_path(receipt['evaluations'][variant])
            protocol = json.loads((folder / 'protocol.json').read_text())
            if protocol['config'] != config:
                raise ValueError('Frozen config mismatch')
            for name, expected in json.loads((folder / 'inputs.json').read_text()).items():
                path = project_path(name)
                if path.is_file() and digest(path) == expected:
                    actual, location = expected, 'unchanged_local_original'
                elif name in archived_names:
                    actual = hashlib.sha256(archive.read(name)).hexdigest()
                    location = 'immutable_approved_archive'
                elif name == protocol['checkpoint'].replace('\\', '/'):
                    actual = digest(project_path(receipt['checkpoints'][variant]))
                    location = 'relocated_verified_checkpoint'
                elif name.startswith('reports/v8_generator_dev/' + prepared.name + '/'):
                    actual = digest(prepared / name.rsplit('/', 1)[-1])
                    location = 'retrieved_original_authorised_protocol'
                elif (runtime_inputs / 'files' / name).is_file():
                    actual = digest(runtime_inputs / 'files' / name)
                    inputs[(runtime_inputs / 'files' / name).relative_to(ROOT).as_posix()] = actual
                    location = 'retrieved_original_runtime_input'
                else:
                    raise ValueError('Unresolved frozen input: ' + name)
                if actual != expected:
                    raise ValueError('Frozen input hash mismatch: ' + name)
                input_checks.append(dict(model=variant, path=name, sha256=actual, verified_against=location))
            requests = pd.read_csv(folder / 'requests.csv')
            table = read_grading_csv(folder / 'per_image.csv')
            complete = json.loads((folder / 'complete.json').read_text())
            chosen = json.loads((folder / 'selection.json').read_text())
            if len(requests) != complete['cases'] or len(table) != len(requests) * 9:
                raise ValueError('Incomplete cases or judge/control rows')
            if requests.case.tolist() != list(range(len(requests))) or requests.duplicated(primary_keys).any():
                raise ValueError('Duplicate or nonsequential request identity')
            if requests.dataset_index.nunique() != len(selected) or requests.patient_id.nunique() != len(selected):
                raise ValueError('Incorrect validation patient coverage')
            expected = set()
            for axis in ('manufacturer', 'field_strength'):
                candidates = settings(config, axis)
                points = [config['evaluation']['fixed_settings'][axis]]
                if chosen[axis]['selected'] is not None and chosen[axis]['selected'] not in points:
                    points.append(chosen[axis]['selected'])
                for seed, points_for_seed, phase in [
                    (config['evaluation']['selection_generation_seed'], candidates, 'selection'),
                    *[(s, points, 'confirmation_same_patients') for s in config['evaluation']['confirmation_generation_seeds']]]:
                    for point in points_for_seed:
                        expected.update((index, axis, point['timestep'], point['guidance'], seed, phase) for index in selected)
                if not same_selection(select_setting(table, axis, config), chosen[axis]):
                    raise ValueError('Saved selection cannot be reproduced')
            actual = set(requests[primary_keys + ['evaluation_phase']].itertuples(index=False, name=None))
            if actual != expected:
                raise ValueError('Missing or unexpected source/setting/seed/phase requests')
            rows = table.set_index(['case', 'judge', 'variant'], verify_integrity=True)
            natives = {}
            for request in requests.itertuples():
                item = val[request.dataset_index]
                if item['patient_id'] != request.patient_id or request.actual_sampling_seed != request.generation_seed + request.dataset_index:
                    raise ValueError('Patient or sampling seed mismatch')
                target_m, target_f = item['manufacturer_label'], item['field_strength_label']
                original = target_m if request.axis == 'manufacturer' else target_f
                if request.axis == 'manufacturer':
                    target_m = 1 - target_m
                else:
                    target_f = 1 - target_f
                if (request.target_manufacturer, request.target_field_strength, request.direction) != (target_m, target_f, f'{original}_to_{1-original}'):
                    raise ValueError('Edited or nonedited acquisition label mismatch')
                support = request_support(train.frame, target_m, target_f, float(item['echo_time_norm']), config['data']['minimum_cell_patients'])
                for key, value in support.items():
                    saved = getattr(request, key)
                    if isinstance(value, float):
                        if not np.isclose(saved, value, rtol=0, atol=1e-12):
                            raise ValueError('Support mismatch: ' + key)
                    elif saved != value:
                        raise ValueError('Support mismatch: ' + key)
                with np.load(folder / request.array_file, allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['source'], val.image_01(request.dataset_index))
                    for echo in (arrays['source_echo_time_norm'].item(), arrays['target_echo_time_norm'].item(),
                                 request.source_echo_time_norm, request.target_echo_time_norm):
                        if not np.isclose(echo, float(item['echo_time_norm']), rtol=0, atol=1e-15):
                            raise ValueError('EchoTime changed')
                    if (arrays['target_manufacturer'].item(), arrays['target_field_strength'].item()) != (target_m, target_f):
                        raise ValueError('Array target mismatch')
                    image_stack = np.stack([arrays[key] for key in ('source', 'null', 'target')])
                if image_stack.shape != (3, 128, 128) or not np.isfinite(image_stack).all() or image_stack.min() < 0 or image_stack.max() > 1:
                    raise ValueError('Invalid saved image values')
                key = tuple(getattr(request, key) for key in primary_keys)
                source_hash = hashlib.sha256(image_stack[0].tobytes()).hexdigest()
                if key in common_sources and common_sources[key] != source_hash:
                    raise ValueError('Models used different source arrays')
                common_sources[key] = source_hash
                natives[request.case] = source_native(image_stack, val.plans[request.dataset_index])
                subset = table[table.case == request.case]
                for j, control in enumerate(('identity', 'null', 'target')):
                    ssim = float(structural_similarity(image_stack[0], image_stack[j], data_range=1., win_size=7))
                    mad = float(np.abs(image_stack[0] - image_stack[j]).mean())
                    scores = subset[subset.variant == control]
                    if not np.allclose(scores.ssim, ssim, rtol=0, atol=1e-9) or not np.allclose(scores.mad, mad, rtol=0, atol=1e-9):
                        raise ValueError('SSIM/MAD recomputation mismatch')
                array_checks.append(dict(model=variant, case=request.case, source_exact=True, fixed_echo=True,
                    labels_support_exact=True, finite_arrays=True, ssim_mad_reproduced=True))
            for seed, path in enumerate(config['evaluation']['judge_paths']):
                judge = load_judge(project_path(path), device).eval().requires_grad_(False)
                name = f'heldout_orientation_robust_seed{seed}'
                error_max = 0.
                for request in requests.itertuples():
                    probabilities = predict_judge(judge, natives[request.case], device)
                    key, other = ('mfg', 'fs') if request.axis == 'manufacturer' else ('fs', 'mfg')
                    target = request.target_manufacturer if key == 'mfg' else request.target_field_strength
                    null_hit = int(probabilities[key][1].argmax() == target)
                    for j, control in enumerate(('identity', 'null', 'target')):
                        saved = rows.loc[(request.case, name, control)]
                        hit = int(probabilities[key][j].argmax() == target)
                        error = abs(float(probabilities[key][j, target]) - saved.target_probability)
                        error_max = max(error_max, error)
                        if error > tolerance or hit != saved.target_hit or hit - null_hit != saved.paired_hit_delta:
                            raise ValueError('Saved target grading mismatch')
                        if int(probabilities[key][0].argmax() != target) != saved.source_correct:
                            raise ValueError('Source correctness mismatch')
                        if int(probabilities[other][j].argmax() == probabilities[other][0].argmax()) != saved.other_axis_stable:
                            raise ValueError('Other-axis grading mismatch')
                grading_checks.append(dict(model=variant, judge=name, cases=len(requests), grades=len(requests)*3,
                    maximum_probability_absolute_error=error_max, discrete_grades_exact=True))
                print('VERIFIED', variant, name, len(requests), 'cases', flush=True)
                del judge
                gc.collect()
            primary = requests
            # Only frozen fixed points are compared between models; selected points may differ.
            mask = np.zeros(len(primary), dtype=bool)
            for axis, point in config['evaluation']['fixed_settings'].items():
                mask |= ((primary.axis == axis) & (primary.timestep == point['timestep']) & (primary.guidance == point['guidance'])).to_numpy()
            keys = set(primary.loc[mask, primary_keys].itertuples(index=False, name=None))
            if common_primary is not None and keys != common_primary:
                raise ValueError('Models have unmatched fixed-point evaluation')
            common_primary = keys
    csv_write(output / 'input_checks.csv', pd.DataFrame(input_checks))
    csv_write(output / 'array_checks.csv', pd.DataFrame(array_checks))
    csv_write(output / 'grading_checks.csv', pd.DataFrame(grading_checks))
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, saved_validation_verified=True,
        cases=len(array_checks), grading_rows=sum(row['grades'] for row in grading_checks),
        matched_fixed_point_requests_per_model=len(common_primary), all_decisions_and_paired_deltas_exact=True,
        full_training_completed=False, converged=False, anatomy_validated=False, test_pixels_used=False))
    print('SAVED VALIDATION VERIFIED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetched', required=True)
    parser.add_argument('--training-fetch', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--runtime-inputs', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    verify(args.fetched, args.training_fetch, args.bundle, args.runtime_inputs, args.run_id, args.device)
