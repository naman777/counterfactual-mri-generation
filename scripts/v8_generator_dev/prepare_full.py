"""Freeze full-training/evaluation rules from previously verified canonical inputs."""
import argparse
import json
import math
import re
import shutil

import pandas as pd

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_protocol import read_config, workload
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs


def validate_baseline(config, old):
    for key in ('view', 'seed', 'model', 'condition', 'diffusion', 'data'):
        if config[key] != old[key]:
            raise ValueError(f'Full protocol is not derived from the verified baseline: {key}')
    current_training, old_training = dict(config['training']), dict(old['training'])
    retries = current_training.pop('amp_gradient_retries', 0)
    old_retries = old_training.pop('amp_gradient_retries', 0)
    if current_training != old_training:
        raise ValueError('Full protocol is not derived from the verified baseline: training')
    return dict(amp_gradient_retries_before=old_retries, amp_gradient_retries_after=retries,
                other_training_settings_unchanged=True)


def prepare(config_path, source_preparation, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid full preparation ID')
    config_path, source = project_path(config_path), project_path(source_preparation)
    config = read_config(config_path)
    if not json.loads((source / 'complete.json').read_text())['complete']:
        raise ValueError('Canonical source preparation is incomplete')
    old = json.loads((source / 'protocol.json').read_text())['config']
    precision_policy = validate_baseline(config, old)
    if config['evaluation']['selection_seed'] != old['smoke']['validation_seed']:
        raise ValueError('Retain the preselected validation sources')
    inputs = json.loads((source / 'inputs.json').read_text())
    verify_inputs(inputs)
    train, val = [CanonicalDataset(config['view'], split) for split in ('train', 'val')]
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new full protocol ID')
    output.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(
        config=config, source_preparation=str(source.relative_to(ROOT)),
        precision_policy=precision_policy,
        scientific_role='Prospective development rules frozen before full training; not a preregistered historical analysis',
        unresolved='Protocol confounding, no independent generated-image anatomy review and no untouched final cohort',
    ))
    (output / 'README.md').write_text(
        '# Full generator training/evaluation protocol\n\n'
        'Producer: `python -m scripts.v8_generator_dev.prepare_full`.\n'
        'protocol.json is the resolved copied configuration, stopping, selection and resource policy. '
        'validation_selection.csv and support_requests.csv retain the preselected validation patients '
        'and every fixed-EchoTime support diagnostic from source preparation. '
        'selection_support.csv counts eligible patients by axis/direction for this exact selection. '
        'resource_plan.json records exact planned steps/model calls and execution caps, not an inferred price. '
        'inputs.json and code/ freeze source/config/judge hashes and implementation. '
        'complete.json records protocol readiness versus unapproved cloud spending and unvalidated anatomy.\n', encoding='utf-8')
    dependencies = [config_path, project_path(config['base_config']), ROOT / 'configs/generator_full_v1.json',
                    source / 'complete.json', source / 'protocol.json']
    dependencies += [project_path(path) for path in config['evaluation']['judge_paths']]
    dependencies += sorted((ROOT / 'scripts/v8_generator_dev').glob('*.py'))
    dependencies += [ROOT / 'scripts/v2_quickwins/judge_model.py', ROOT / 'scripts/v5_anatomy/metrics.py']
    for path in dependencies:
        relative = path.relative_to(ROOT)
        inputs[relative.as_posix()] = digest(path)
        if path.suffix in {'.py', '.json'}:
            target = output / 'code' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for name in ('validation_selection.csv', 'support_requests.csv'):
        path = source / name
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        shutil.copyfile(path, output / name)
    selected = pd.read_csv(output / 'validation_selection.csv')
    requests = pd.read_csv(output / 'support_requests.csv')
    requests = requests[requests.sop_uid.isin(selected.sop_uid)].copy()
    requests['direction'] = requests.apply(lambda row: f'{1 - int(row.target_manufacturer if row.axis == "manufacturer" else row.target_field_strength)}_to_{int(row.target_manufacturer if row.axis == "manufacturer" else row.target_field_strength)}', axis=1)
    records = []
    for axis in ('manufacturer', 'field_strength'):
        for direction in ('0_to_1', '1_to_0'):
            group = requests[(requests.axis == axis) & (requests.direction == direction)]
            eligible = group[group.condition_support_status.isin(config['evaluation']['selection_support_statuses'])]
            records.append(dict(axis=axis, direction=direction, selected_patients=int(group.patient_id.nunique()),
                                eligible_patients=int(eligible.patient_id.nunique()),
                                direction_coverage_met=bool(eligible.patient_id.nunique() >= config['evaluation']['minimum_eligible_patients_per_direction']),
                                exact_observed_patients=int(group.loc[group.condition_support_status == 'observed_condition', 'patient_id'].nunique()),
                                anatomy_validated=False))
    csv_write(output / 'selection_support.csv', pd.DataFrame(records))
    batches = math.ceil(len(train) / config['full_training']['batch_size'])
    plan = dict(
        training_slices=len(train), validation_slices=len(val), validation_patients=len(selected),
        batches_per_epoch_per_variant=batches,
        maximum_training_batches_per_variant=batches * config['full_training']['max_epochs'],
        maximum_saved_blocks_per_variant=math.ceil(config['full_training']['max_epochs'] / config['full_training']['validation_every_epochs']),
        **workload(config, len(selected)), execution_caps=config['resources'],
        source_upload_required_for_cloud=True, cloud_upload_launched=False,
        cost_estimate=None, cost_limit_usd=None,
        monetary_status='Unpriced/unapproved; local engineering allowed, substantial paid execution not launched',
    )
    json_write(output / 'resource_plan.json', plan)
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(
        complete=True, protocol_frozen=True, source_hashes_unchanged=True,
        validation_source_selection_unchanged=True,
        direction_support=pd.DataFrame(records).to_dict('records'),
        resource_caps_are_not_monetary_approval=True, cloud_launched=False,
        efficacy_training_completed=False, anatomy_validated=False, test_pixels_used=False,
    ))
    print('FULL PROTOCOL PREPARED', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/generator_full_v2.json')
    parser.add_argument('--source-preparation', default='reports/v8_generator_dev/preparation_20261116_01')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    prepare(args.config, args.source_preparation, args.run_id)
