"""Audit retrieved paired checkpoint state and save the actual regression-test log."""
import argparse
import json
import math
import re
import subprocess
import sys

import torch

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit(fetched, run_id):
    torch.set_num_threads(2)
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new audit ID required')
    fetched = project_path(fetched)
    result = json.loads((fetched / 'complete.json').read_text())
    if not result['checkpoint_hashes_verified'] or result['full_training_completed']:
        raise ValueError('Expected verified numerically interrupted checkpoints')
    protocol_path, = (fetched / 'root').glob('*/protocol.json')
    record = json.loads(protocol_path.read_text())['selection']['checkpoint_record']
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    inputs, states = {}, {}
    for variant, relative in result['checkpoints'].items():
        path = project_path(relative)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        if digest(path) != record['checkpoint_hashes'][variant]:
            raise ValueError('Checkpoint hash mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        keys = ('conditioning_state_dict', 'unet_state_dict', 'raw_conditioning_state_dict',
                'raw_unet_state_dict', 'optimizer_state_dict', 'scaler_state_dict')
        checks = {key: finite_state(checkpoint[key]) for key in keys}
        if not all(checks.values()) or checkpoint['progress'] != record['progress'][variant]:
            raise ValueError('Nonfinite or unmatched saved state')
        states[variant] = dict(checks=checks, progress=checkpoint['progress'],
                               training_torch=checkpoint['torch_version'], local_torch=str(torch.__version__),
                               scaler=checkpoint['scaler_state_dict'],
                               mixed_precision=checkpoint['config']['training']['mixed_precision'])
        del checkpoint
    json_write(output / 'checkpoint_state.json', states)
    modules = ['test_pipeline', 'test_full', 'test_analysis', 'test_guides',
               'test_resource_limited', 'test_verification', 'test_failed_pair', 'test_precision_step']
    command = [sys.executable, '-m', 'unittest',
               *[f'scripts.v8_generator_dev.{name}' for name in modules], '-v']
    with (output / 'tests.txt').open('x', encoding='utf-8') as handle:
        tests = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    json_write(output / 'tests.json', dict(command=command, exit_code=tests.returncode))
    for path in sorted((ROOT / 'scripts/v8_generator_dev').glob('*.py')):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    config_path = ROOT / 'configs/generator_full_v2_precision.json'
    inputs[config_path.relative_to(ROOT).as_posix()] = digest(config_path)
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=tests.returncode == 0,
                                              saved_checkpoint_state_finite=True, full_training_completed=False,
                                              numerical_failure_reproduced=False, numerical_issue_fixed=False,
                                              exact_cross_version_resume_claimed=False, anatomy_validated=False))
    (output / 'README.md').write_text(
        '# Retrieved checkpoint audit\n\nProducer: `python -m scripts.v8_generator_dev.audit_failed_pair`.\n'
        'checkpoint_state.json records actual raw/EMA/optimizer/scaler finiteness, progress and runtime versions. '
        'tests.txt/tests.json retain the executed regression suite. inputs.json hashes unchanged checkpoints; '
        'complete.json records software checks only. This does not reproduce/fix the later numerical failure, '
        'establish editing efficacy or certify cross-version training resume.\n', encoding='utf-8')
    print('CHECKPOINT AUDIT', output, 'tests_exit', tests.returncode, flush=True)
    if tests.returncode:
        raise RuntimeError('Regression suite failed; retained the failed audit')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetched', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    audit(args.fetched, args.run_id)
