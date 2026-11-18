"""Verify saved recovery/grading with literal null controls and matched CUDA settings."""
import argparse
import gc
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs


def assert_equal(a, b, name='state'):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError(f'Interrupted/direct tensors differ: {name}')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f'State keys differ: {name}')
        for key in a:
            assert_equal(a[key], b[key], f'{name}.{key}')
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'State lengths differ: {name}')
        for index, (x, y) in enumerate(zip(a, b)):
            assert_equal(x, y, f'{name}.{index}')
    elif a != b:
        raise AssertionError(f'States differ: {name}')


def verify(prepared, first, resumed, direct, evaluations, run_id, device='cuda'):
    from scripts.v2_quickwins.judge_model import load_judge
    deterministic_runtime()
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid verification ID')
    paths = [project_path(value) for value in (prepared, first, resumed, direct, *evaluations)]
    prepared, first, resumed, direct, *evaluations = paths
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new verification ID')
    output.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(inputs=[str(p.relative_to(ROOT)) for p in paths],
                                              role='Engineering reproducibility and validation integration verification'))
    (output / 'README.md').write_text(
        '# Full training/validation engineering verification\n\n'
        'Producer: `python -m scripts.v8_generator_dev.verify_full_v2`.\n'
        'protocol.json identifies completed preparations/segments/evaluations. '
        'unit_tests.txt and unit_tests.json record the actual test commands/result. '
        'resume_checks.csv records bitwise equality of interrupted/direct training states and draws. '
        'grading_checks.csv records recomputation of saved judge probabilities/hits and canonical sources. '
        'inputs.json and code/ retain hashes/source snapshot. complete.json records passed checks '
        'and the explicit absence of full efficacy or independent anatomy validation.\n', encoding='utf-8')
    inputs = {}
    for folder in paths:
        if not json.loads((folder / 'complete.json').read_text())['complete']:
            raise ValueError('Input stage incomplete')
        verify_inputs(json.loads((folder / 'inputs.json').read_text()))
        for path in folder.rglob('*'):
            if path.is_file():
                inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    command = [sys.executable, '-m', 'unittest', 'scripts.v8_generator_dev.test_pipeline', 'scripts.v8_generator_dev.test_full', 'scripts.v8_generator_dev.test_verification', '-v']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    with (output / 'unit_tests.txt').open('x', encoding='utf-8') as handle:
        handle.write(result.stdout + result.stderr)
    json_write(output / 'unit_tests.json', dict(command=command, exit_code=result.returncode))
    if result.returncode:
        raise AssertionError('Unit checks failed')
    checkpoints = [json.loads((folder / 'complete.json').read_text())['checkpoints'] for folder in (resumed, direct)]
    checks = []
    for variant in ('additive', 'scale_shift'):
        pair = []
        for mapping in checkpoints:
            path = project_path(mapping[variant])
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            pair.append(torch.load(path, map_location='cpu', weights_only=False))
        for key in ('config', 'variant', 'protocol_sha256', 'data_identity', 'effective_batch_size',
                    'progress', 'history', 'conditioning_state_dict', 'unet_state_dict',
                    'raw_conditioning_state_dict', 'raw_unet_state_dict', 'optimizer_state_dict',
                    'scaler_state_dict', 'rng_torch', 'rng_cuda'):
            assert_equal(pair[0][key], pair[1][key], key)
        rows = []
        for folder in (first, resumed):
            rows.extend(json.loads((folder / 'block_0001' / f'{variant}_steps.json').read_text()))
        reference = json.loads((direct / 'block_0001' / f'{variant}_steps.json').read_text())
        if len(rows) != len(reference):
            raise AssertionError('Interrupted/direct batch counts differ')
        for actual, expected in zip(rows, reference):
            for key in ('global_step', 'epoch_at_draw', 'sample_indices', 'patient_ids', 'timesteps',
                        'noise_sha256', 'noise_mse', 'grad_norm', 'applied'):
                assert_equal(actual[key], expected[key], key)
        checks.append(dict(variant=variant, compared_steps=len(rows), raw_weights_bitwise=True,
                           ema_bitwise=True, optimizer_scaler_bitwise=True, rng_cursor_bitwise=True,
                           sampled_sources_noise_losses_bitwise=True))
        del pair
        gc.collect()
    csv_write(output / 'resume_checks.csv', pd.DataFrame(checks))
    config = json.loads((prepared / 'protocol.json').read_text())['config']
    dataset = CanonicalDataset(config['view'], 'val')
    grading = []
    for folder in evaluations:
        table = read_grading_csv(folder / 'per_image.csv')
        requests = pd.read_csv(folder / 'requests.csv')
        for request in requests.itertuples():
            with np.load(folder / request.array_file, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['source'], dataset.image_01(request.dataset_index))
                np.testing.assert_array_equal(arrays['source_echo_time_norm'], arrays['target_echo_time_norm'])
        for seed, path in enumerate(config['evaluation']['judge_paths']):
            path = project_path(path)
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            judge = load_judge(path, device).eval().requires_grad_(False)
            name = f'heldout_orientation_robust_seed{seed}'
            checked = 0
            for request in requests.itertuples():
                with np.load(folder / request.array_file, allow_pickle=False) as arrays:
                    native = source_native(np.stack([arrays[name] for name in ('source', 'null', 'target')]),
                                           dataset.plans[request.dataset_index])
                probs = predict_judge(judge, native, device)
                key = 'mfg' if request.axis == 'manufacturer' else 'fs'
                target = request.target_manufacturer if key == 'mfg' else request.target_field_strength
                subset = table[(table.case == request.case) & (table.judge == name)]
                null_hit = int(probs[key][1].argmax() == target)
                for j, variant in enumerate(('identity', 'null', 'target')):
                    row = subset[subset.variant == variant].iloc[0]
                    hit = int(probs[key][j].argmax() == target)
                    if hit != row.target_hit or not np.isclose(probs[key][j, target], row.target_probability, rtol=0, atol=1e-7):
                        raise AssertionError('Saved probability/hit does not reproduce')
                    if hit - null_hit != row.paired_hit_delta:
                        raise AssertionError('Saved paired target/null delta is wrong')
                    checked += 1
            grading.append(dict(evaluation=folder.name, judge=name, rows_recomputed=checked,
                                probabilities_reproduce=True, paired_hit_deltas_reproduce=True,
                                source_native_frame_verified=True, echo_time_fixed=True))
            del judge
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        chosen = json.loads((folder / 'selection.json').read_text())
        if any(value['selected'] is not None for value in chosen.values()):
            raise AssertionError('Engineering evaluation selected an operating point')
    csv_write(output / 'grading_checks.csv', pd.DataFrame(grading))
    for path in (ROOT / 'scripts/v8_generator_dev').glob('*.py'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        target = output / 'code' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(
        complete=True, unit_tests_passed=True, interrupted_direct_bitwise=True,
        actual_mri_model_resume_verified=True, all_saved_grading_recomputed=True,
        source_inputs_unchanged=True, full_training_completed=False,
        operating_point_selected=False, anatomy_validated=False,
        test_pixels_used=False, cloud_launched=False,
    ))
    print('FULL ENGINEERING VERIFICATION COMPLETE', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('prepared', 'first', 'resumed', 'direct', 'run-id'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--evaluation', action='append', default=[])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    verify(args.prepared, args.first, args.resumed, args.direct, args.evaluation, args.run_id, args.device)
