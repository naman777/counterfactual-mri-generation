"""Save stage-specific tests and verify frozen source/checkpoint/array provenance."""
import argparse
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs


def verify(prepared, smoke, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid verification ID')
    prepared, smoke = project_path(prepared), project_path(smoke)
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new verification ID')
    output.mkdir(parents=True)
    (output / 'README.md').write_text(
        '# Generator development stage verification\n\n'
        'Producer: `python -m scripts.v8_generator_dev.verify_stage`.\n'
        'protocol.json identifies the verified preparation and smoke. '
        'unit_tests.txt and unit_tests.json record the actual unittest invocation/result. '
        'artifact_checks.csv records checkpoint/array comparisons. '
        'inputs.json fingerprints verified artifacts and code; code/ snapshots verification code. '
        'complete.json records the final engineering checks and unresolved efficacy/anatomy limits.\n'
        'This verifies software/provenance, not acquisition-edit or anatomical performance.\n', encoding='utf-8')
    json_write(output / 'protocol.json', dict(prepared=str(prepared.relative_to(ROOT)),
                                              smoke=str(smoke.relative_to(ROOT)),
                                              role='Read-only verification of completed immutable reports'))
    inputs = {}
    for folder in (prepared, smoke):
        if not json.loads((folder / 'complete.json').read_text())['complete']:
            raise ValueError('Input stage incomplete')
        verify_inputs(json.loads((folder / 'inputs.json').read_text()))
        for path in folder.rglob('*'):
            if path.is_file():
                inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    # Run from the project root; no original test images are used by synthetic tests.
    command = [sys.executable, '-m', 'unittest', 'scripts.v8_generator_dev.test_pipeline', '-v']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    with (output / 'unit_tests.txt').open('x', encoding='utf-8') as handle:
        handle.write(result.stdout + result.stderr)
    json_write(output / 'unit_tests.json', dict(command=command, exit_code=result.returncode))
    if result.returncode:
        raise AssertionError('Stage tests failed; see unit_tests.txt')
    summary = json.loads((smoke / 'complete.json').read_text())
    config = json.loads((prepared / 'protocol.json').read_text())['config']
    dataset = CanonicalDataset(config['view'], 'val')
    selected = pd.read_csv(prepared / 'validation_selection.csv')
    indices = selected.dataset_index.astype(int).tolist()[:config['smoke']['inference_patients']]
    expected_source = np.stack([dataset.image_01(i) for i in indices])[:, None]
    expected_echo = dataset.frame.iloc[indices].echo_time_norm.to_numpy(np.float32)
    checks = []
    draws = []
    for variant, run in zip(config['variants'], summary['variants']):
        if run['variant'] != variant or not run['checkpoint_roundtrip_bitwise'] or not run['seeded_inference_bitwise']:
            raise AssertionError('Variant/checkpoint/inference metadata mismatch')
        checkpoint = project_path(run['checkpoint'])
        if digest(checkpoint) != run['checkpoint_sha256']:
            raise ValueError('Smoke checkpoint changed')
        inputs[checkpoint.relative_to(ROOT).as_posix()] = run['checkpoint_sha256']
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['format'] != 'canonical_generator_dev_v1' or ck['variant'] != variant or ck['config'] != config:
            raise AssertionError('Incorrect canonical checkpoint metadata')
        if ck['protocol_sha256'] != digest(smoke / 'protocol.json'):
            raise AssertionError('Checkpoint/protocol hash mismatch')
        for family, raw_key, ema_key in [('unet', 'raw_unet_state_dict', 'unet_state_dict'),
                                         ('conditioner', 'raw_conditioning_state_dict', 'conditioning_state_dict')]:
            if not any(not torch.equal(ck[raw_key][key], value) for key, value in ck[ema_key].items()):
                raise AssertionError('Raw and EMA states were not separately retained')
        draws.append(pd.read_csv(smoke / f'{variant}_draws.csv'))
        with np.load(smoke / f'{variant}_wiring_arrays.npz', allow_pickle=False) as arrays:
            # Conversion [-1,1] -> [0,1] incurs a single-precision roundoff.
            np.testing.assert_allclose(arrays['source'], expected_source, rtol=0, atol=3e-8)
            for name in ('source', 'null', 'manufacturer', 'field_strength'):
                np.testing.assert_array_equal(arrays[f'{name}_echo_time_norm'], expected_echo)
                if not np.isfinite(arrays[name]).all() or arrays[name].min() < 0 or arrays[name].max() > 1:
                    raise AssertionError('Invalid saved wiring array')
            for axis in ('manufacturer', 'field_strength'):
                difference = float(np.abs(arrays[axis] - arrays['null']).mean())
                if difference == 0:
                    raise AssertionError('Changed acquisition condition has no inference effect')
                checks.append(dict(variant=variant, axis=axis, patients=len(indices),
                                   mean_absolute_target_null_difference=difference,
                                   source_array_matches=True, source_echo_time_fixed=True,
                                   checkpoint_hash_matches=True, raw_and_ema_separate=True,
                                   interpretation='Nonzero wiring response only; no target recognition/anatomy claim'))
        del ck
    pd.testing.assert_frame_equal(draws[0], draws[1])
    csv_write(output / 'artifact_checks.csv', pd.DataFrame(checks))
    for path in sorted((ROOT / 'scripts/v8_generator_dev').glob('*.py')):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        target = output / 'code' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(
        complete=True, unit_tests_passed=True, unit_tests_exit_code=result.returncode,
        source_inputs_unchanged=True, checkpoint_hashes_verified=True,
        canonical_sources_match_wiring_arrays=True, raw_and_ema_separate=True,
        matched_sources_timesteps_noise=True, fixed_source_echo_time=True,
        all_condition_branches_nonzero=True, test_pixels_used=False, cloud_used=False,
        efficacy_established=False, anatomy_validated=False, winner_selected=False,
    ))
    print('VERIFIED', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--smoke', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    verify(args.prepared, args.smoke, args.run_id)
