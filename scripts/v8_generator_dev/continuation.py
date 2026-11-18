"""Explicit resource authorisation and same-runtime matched precision migration."""
import argparse
import copy
import json
import re
import shutil

import torch

from scripts.v8_generator_dev.full_protocol import validate_full_config
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.prepare_precision_resume import migrate_payload, resumed_config, training_state_equal


def proposed_config(original):
    config = resumed_config(original)
    config['resources']['max_variant_wall_seconds'] = 10800
    config['resources']['maximum_cloud_worker_seconds'] = 13500
    config['resources']['max_validation_wall_seconds'] = 2100
    validate_full_config(config)
    return config


def authorised_config(original, proposed, budget):
    if proposed != proposed_config(original):
        raise ValueError('Proposed continuation changes undeclared scientific settings')
    if budget != 6.0:
        raise ValueError('This prepared substantial continuation requires a new explicit US$6 allowance')
    config = copy.deepcopy(proposed)
    config['resources'].update(monetary_budget_authorised=True, authorised_allowance_usd=budget,
        cost_status='New explicit continuation allowance; no reuse or reset of the original allowance')
    return config


def continuation_payload(original, config, protocol_hash, parent_hash, budget):
    expected = authorised_config(original['config'], proposed_config(original['config']), budget)
    if config != expected:
        raise ValueError('Only declared precision/resource changes may be rebound')
    payload = migrate_payload(original, resumed_config(original['config']), protocol_hash, parent_hash)
    payload['config'] = copy.deepcopy(config)
    payload['continuation_resources'] = dict(new_approved_allowance_usd=budget,
        original_experiment_allowance_reset=False, scientific_change='Bounded same-draw AMP retry only')
    if not training_state_equal(original, payload):
        raise AssertionError('Rebinding changed retained training state')
    return payload


def materialise(plan_path, run_id, budget):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new run ID required')
    plan_path = project_path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan['run_id'] != run_id:
        raise ValueError('Run ID differs from frozen proposal')
    source = project_path('reports/v8_generator_dev/full_20261116_04_protocol')
    original = json.loads((source / 'protocol.json').read_text())['config']
    config = authorised_config(original, plan['proposed_config'], budget)
    inputs = json.loads((source / 'inputs.json').read_text())
    # Worker verifies original bytes before applying the explicitly frozen code patch.
    for name, expected in plan['patch_hashes'].items():
        path = ROOT / 'scripts/v8_generator_dev' / name
        if digest(path) != expected:
            raise ValueError('Frozen continuation source changed: '+name)
        inputs[path.relative_to(ROOT).as_posix()] = expected
    output = ROOT / 'reports/v8_generator_dev' / (run_id+'_protocol')
    ck_dir = ROOT / 'checkpoints/diffusion/ax_t1_128_v8_generator_dev' / (run_id+'_resume')
    output.mkdir(exist_ok=False)
    ck_dir.mkdir(exist_ok=False)
    json_write(output / 'protocol.json', dict(config=config, parent_protocol_sha256=digest(source / 'protocol.json'),
        source_preparation='reports/v8_generator_dev/preparation_20261116_01',
        scientific_role='Explicit matched epoch-70 continuation; precision policy changed, original stopping/evaluation retained',
        original_run='full_20261116_04', new_approved_allowance_usd=budget,
        test_pixels_used=False, expert_anatomy_validated=False))
    protocol_hash = digest(output / 'protocol.json')
    shutil.copyfile(source / 'validation_selection.csv', output / 'validation_selection.csv')
    pair = copy.deepcopy(plan['selection']['checkpoint_record'])
    checkpoints, hashes, checks, progress = {}, {}, {}, {}
    for variant, relative in pair['checkpoints'].items():
        parent = project_path(relative)
        parent_hash = digest(parent)
        if parent_hash != pair['checkpoint_hashes'][variant]:
            raise ValueError('Original matched parent hash mismatch')
        old = torch.load(parent, map_location='cpu', weights_only=False)
        if old['config'] != original or old['progress'] != pair['progress'][variant] or old['variant'] != variant:
            raise ValueError('Parent identity/exposure mismatch')
        if old['torch_version'] != str(torch.__version__) or old['device_type'] != 'cuda':
            raise ValueError('Migration must use the original CUDA package runtime')
        payload = continuation_payload(old, config, protocol_hash, parent_hash, budget)
        destination = ck_dir / f'{variant}.pt'
        with destination.open('xb') as handle:
            torch.save(payload, handle)
        restored = torch.load(destination, map_location='cpu', weights_only=False)
        if not training_state_equal(old, restored):
            raise AssertionError('Round-trip state mismatch')
        checkpoints[variant], hashes[variant] = str(destination.relative_to(ROOT)), digest(destination)
        checks[variant] = dict(retained_training_state_bitwise_equal=True, original_torch_runtime=True,
            parent_sha256=parent_hash, migrated_sha256=hashes[variant])
        progress[variant] = old['progress']
        inputs[relative.replace('\\', '/')] = parent_hash
        del old, payload, restored
    if progress['additive'] != progress['scale_shift']:
        raise ValueError('Continuation must start from a matched pair')
    pair.update(checkpoints=checkpoints, checkpoint_hashes=hashes, progress=progress, prepared_sha256=protocol_hash)
    json_write(output / 'prior_pair.json', pair)
    json_write(output / 'migration_checks.json', checks)
    inputs[plan_path.relative_to(ROOT).as_posix()] = digest(plan_path)
    inputs[(source / 'protocol.json').relative_to(ROOT).as_posix()] = digest(source / 'protocol.json')
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, migration_completed=True,
        original_runtime_retained=True, matched_epoch70=True, full_training_completed=False))
    (output / 'README.md').write_text(
        '# Authorised matched continuation protocol\n\nProducer: `python -m scripts.v8_generator_dev.continuation`.\n'
        'protocol.json freezes scientific/resource settings; validation_selection.csv retains original sources. '
        'prior_pair.json maps matched migrated checkpoint copies; migration_checks.json checks all retained state. '
        'inputs.json hashes unchanged data/parents and explicitly patched code. complete.json describes migration '
        'only, never training completion, clinical preservation or editing success.\n', encoding='utf-8')
    print('CONTINUATION MATERIALISED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--budget-usd', type=float, required=True)
    args = parser.parse_args()
    materialise(args.plan, args.run_id, args.budget_usd)
