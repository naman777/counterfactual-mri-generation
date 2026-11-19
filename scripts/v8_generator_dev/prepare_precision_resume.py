"""Explicit precision-only state migration; strict trainer resume checks stay unchanged."""
import argparse
import copy
import json
import re
import shutil
import subprocess
import sys

import torch

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.numerical_replay import equal_state


def resumed_config(original):
    candidate = copy.deepcopy(original)
    if candidate['training'].get('amp_gradient_retries', 0) != 0:
        raise ValueError('Expected the original zero-retry protocol')
    candidate['training']['amp_gradient_retries'] = 8
    candidate['resources']['monetary_budget_authorised'] = False
    candidate['resources']['authorised_allowance_usd'] = 0.0
    candidate['resources']['cost_status'] = 'Unapproved efficacy continuation; diagnostic approval is not a new full-training allowance'
    return candidate


def migrate_payload(original, candidate_config, protocol_hash, parent_sha256):
    if candidate_config != resumed_config(original['config']):
        raise ValueError('Only declared precision policy and explicit unapproved budget metadata may change')
    migrated = copy.deepcopy(original)
    migrated['config'] = copy.deepcopy(candidate_config)
    migrated['protocol_sha256'] = protocol_hash
    migrated['precision_migration'] = dict(format='precision_only_resume_v1', parent_sha256=parent_sha256,
        amp_gradient_retries_before=0, amp_gradient_retries_after=8, optimiser_reset=False,
        scaler_reset=False, raw_ema_rng_cursor_retained=True, full_training_completed=False)
    return migrated


def training_state_equal(original, migrated):
    retained_keys = set(original) - {'config', 'protocol_sha256'}
    return all(equal_state(original[key], migrated[key]) for key in retained_keys)


def prepare(analysis, checkpoint_fetch, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new preparation ID required')
    analysis, checkpoint_fetch = map(project_path, (analysis, checkpoint_fetch))
    diagnostic = json.loads((analysis / 'complete.json').read_text())
    if not diagnostic['controlled_retry_gate_passed']:
        raise ValueError('Recovered-failure retry control must pass before migration preparation')
    verify_inputs(json.loads((analysis / 'inputs.json').read_text()))
    fetched = json.loads((checkpoint_fetch / 'complete.json').read_text())
    if fetched['common_epoch'] != 70 or not fetched['checkpoint_hashes_verified']:
        raise ValueError('Verified matched epoch-70 pair required')
    base = ROOT / 'reports/v8_generator_dev'
    source_prepared = base / 'precision_preparation_20261117_02'
    inputs = json.loads((source_prepared / 'inputs.json').read_text())
    verify_inputs(inputs)
    parent_protocol = base / 'cloud_resume_fetch_20261116_01/training/protocol.json'
    original_config = json.loads(parent_protocol.read_text())['config']
    config = resumed_config(original_config)
    output = base / run_id
    checkpoint_output = ROOT / 'checkpoints/diffusion/ax_t1_128_v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    checkpoint_output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Explicit precision-only resume preparation\n\nProducer: `python -m scripts.v8_generator_dev.prepare_precision_resume`.\n'
        'config.json/protocol.json freeze unchanged scientific settings except bounded AMP retry, with '
        'efficacy spending explicitly unauthorised. validation_selection.csv retains identical sources. '
        'prior_pair.json maps new, matched epoch-70 checkpoint copies. migration_checks.json verifies all '
        'raw/EMA/optimizer/scaler/RNG/cursor/history state after round-trip loading. tests.txt/tests.json '
        'verify the original strict resume rejection and explicit migrated resume. code/ snapshots this '
        'migration implementation. inputs.json hashes unchanged data/code/parents. complete.json means '
        'preparation, not a launched, stable or complete efficacy training run.\n', encoding='utf-8')
    json_write(output / 'config.json', config)
    json_write(output / 'protocol.json', dict(config=config,
        parent_training_protocol=str(parent_protocol.relative_to(ROOT)),
        parent_training_protocol_sha256=digest(parent_protocol),
        diagnostic_analysis=str(analysis.relative_to(ROOT)),
        scientific_change='Only amp_gradient_retries: 0 to 8; all sampling/loss/learning-rate/stopping/evaluation settings retained',
        checkpoint_role='Last matched epoch-70 pair; not the later unmatched additive failure snapshot',
        resource_status='Prepared only; no full efficacy continuation or new budget is authorised',
        original_gpu_runtime_required=True, source_preparation='reports/v8_generator_dev/preparation_20261116_01'))
    shutil.copyfile(source_prepared / 'validation_selection.csv', output / 'validation_selection.csv')
    protocol_hash = digest(output / 'protocol.json')
    parents, checkpoints, hashes, checks, progress = {}, {}, {}, {}, {}
    for variant in ('additive', 'scale_shift'):
        path = project_path(fetched['checkpoints'][variant])
        original = torch.load(path, map_location='cpu', weights_only=False)
        if original['config'] != original_config or original['variant'] != variant or original['progress']['epoch'] != 70:
            raise ValueError('Parent checkpoint config/variant/exposure mismatch')
        destination = checkpoint_output / f'{variant}.pt'
        payload = migrate_payload(original, config, protocol_hash, digest(path))
        with destination.open('xb') as handle:
            torch.save(payload, handle)
        roundtrip = torch.load(destination, map_location='cpu', weights_only=False)
        if not training_state_equal(original, roundtrip):
            raise AssertionError('Migration changed a retained training state')
        parents[variant] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
        checkpoints[variant] = str(destination.relative_to(ROOT))
        hashes[variant] = digest(destination)
        checks[variant] = dict(all_retained_training_state_bitwise_equal=True,
            parent_torch=original['torch_version'], parent_device=original['device_type'],
            original_scaler_retained=True, original_progress_retained=True,
            config_change_explicit=True, runtime_migration_claimed=False)
        progress[variant] = original['progress']
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        del original, payload, roundtrip
    if progress['additive'] != progress['scale_shift'] or progress['additive']['sample_offset'] != 0:
        raise ValueError('Migrated pair must retain fully matched complete-epoch exposure')
    json_write(output / 'prior_pair.json', dict(checkpoints=checkpoints, checkpoint_hashes=hashes,
        progress=progress, prepared_sha256=protocol_hash, engineering=False,
        matched_sources_timesteps_noise=True, checkpoint_bytes=sum(p.stat().st_size for p in checkpoint_output.iterdir()),
        parents=parents, full_training_completed=False))
    json_write(output / 'migration_checks.json', checks)
    modules = ['test_pipeline', 'test_full', 'test_analysis', 'test_guides', 'test_resource_limited',
        'test_verification', 'test_failed_pair', 'test_precision_step', 'test_numerical_replay', 'test_precision_resume']
    command = [sys.executable, '-m', 'unittest', *['scripts.v8_generator_dev.'+name for name in modules], '-v']
    with (output / 'tests.txt').open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
    json_write(output / 'tests.json', dict(command=command, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError('Resume tests failed; preserve unpromoted preparation')
    (output / 'code').mkdir()
    for name in ('prepare_precision_resume.py', 'test_precision_resume.py'):
        path = ROOT / 'scripts/v8_generator_dev' / name
        shutil.copyfile(path, output / 'code' / name)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    for path in (analysis / 'complete.json', parent_protocol, checkpoint_fetch / 'complete.json'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, precision_resume_prepared=True,
        common_epoch=70, all_retained_training_state_bitwise_equal=True,
        actual_cloud_resume_tested=False, cross_version_resume_claimed=False,
        full_training_launched=False, full_training_completed=False,
        monetary_budget_authorised=False, anatomy_validated=False))
    print('PRECISION RESUME PREPARED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--checkpoint-fetch', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    prepare(args.analysis, args.checkpoint_fetch, args.run_id)
