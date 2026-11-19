"""Interpret recovered numerical controls, retaining limited training scope and budget."""
import argparse
import json
import re

import torch

from scripts.v8_generator_dev.audit_failed_pair import finite_state
from scripts.v8_generator_dev.diagnostic_workflow import latest_ledger
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs


def analyse(fetched, support, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new report ID required')
    fetched, support = map(project_path, (fetched, support))
    verify_inputs(json.loads((fetched / 'inputs.json').read_text()))
    verify_inputs(json.loads((support / 'inputs.json').read_text()))
    receipt = json.loads((fetched / 'complete.json').read_text())
    if not receipt['complete']:
        raise ValueError('Incomplete diagnostic must retain partial status')
    home = project_path(receipt['home'])
    result = json.loads((home / 'results/complete.json').read_text())
    worker = json.loads((home / 'complete.json').read_text())
    if worker['diagnostic'] != result or result['full_training_completed'] or result['new_efficacy_checkpoint']:
        raise ValueError('Diagnostic scope/receipt mismatch')
    plan = json.loads((home / 'protocol.json').read_text())
    if plan['original_approval_usd'] != 6 or plan['new_allowance']:
        raise ValueError('Spending must remain within original approval')
    support_result = json.loads((support / 'complete.json').read_text())
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Numerical diagnostic analysis\n\nProducer: `python -m scripts.v8_generator_dev.analyse_numerical_diagnostic`.\n'
        'RESULTS.md interprets recovered controls and support limits. snapshot_checks.json audits the '
        'recovered pre-step state; budget.json debits all diagnostic and historical launch durations. '
        'inputs.json binds raw receipts, source and metadata audit. complete.json closes only the '
        'diagnostic milestone, never full training, efficacy or anatomy validation.\n', encoding='utf-8')
    inputs = {path.relative_to(ROOT).as_posix(): digest(path) for path in home.rglob('*') if path.is_file()}
    original_protocol = ROOT / 'reports/v8_generator_dev/cloud_resume_fetch_20261116_01/training/protocol.json'
    duration_receipts = list((ROOT / 'reports/v8_generator_dev').glob('diagnostic_launch_*/process_result.json'))
    for path in (support / 'complete.json', support / 'direction_coverage.csv', support / 'inputs.json',
                 original_protocol, ROOT / 'scripts/v8_generator_dev/analyse_numerical_diagnostic.py',
                 ROOT / 'scripts/v8_generator_dev/diagnostic_workflow.py',
                 ROOT / 'reports/v8_generator_dev/failed_pair_report_20261117_01/budget.json', *duration_receipts):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    recovered, remedy = result['failure_recovered'], False
    lines = ['# Failed-block numerical diagnostic', '',
        'This is a bounded reproduction and single-update control experiment using the original '
        'epoch-70 additive checkpoint, historical engine, package pins and A10 GPU class. '
        'The historical failing batch and driver identity were not retained; matching this setup '
        'does not prove a bitwise identical historical trajectory.', '']
    if recovered:
        failure = json.loads((home / 'results/failure.json').read_text())
        replay = json.loads((home / 'results/replay.json').read_text())
        snapshot = torch.load(home / 'results/pre_step.pt', map_location='cpu', weights_only=False)
        keys = ['conditioning_state_dict', 'unet_state_dict', 'raw_conditioning_state_dict',
                'raw_unet_state_dict', 'optimizer_state_dict', 'scaler_state_dict']
        checks = {key: finite_state(snapshot[key]) for key in keys}
        checks['progress_matches_failure_pre_step'] = snapshot['progress'] == failure['progress_before']
        checks['scaler_matches_failure_pre_step'] = snapshot['scaler_state_dict'] == failure['scaler_before']
        checks['original_config_retained'] = snapshot['config'] == json.loads(original_protocol.read_text())['config']
        checks['original_protocol_hash_retained'] = snapshot['protocol_sha256'] == plan['selection']['checkpoint_record']['prepared_sha256']
        checks['additive_variant_retained'] = snapshot['variant'] == 'additive'
        checks['retry_uses_recovered_failure_draw'] = replay.get('retry_amp', {}).get('draw') == failure['draw']
        if not all(checks.values()):
            raise ValueError('Recovered pre-step state is nonfinite or inconsistent')
        json_write(output / 'snapshot_checks.json', checks)
        remedy = bool(result['replay_control_failed'] and
                      replay['original_control'].get('error') == 'Nonfinite full-training gradient' and result['retry_finite'] and
                      result['safe_update_matches_bitwise'] and result['fp32_finite'] and
                      replay['same_draw_for_successful_modes'])
        lines += [f'A nonfinite-gradient failure recurred during the block replay at epoch '
            f'{failure["progress_before"]["epoch"]}, before global step '
            f'{failure["progress_before"]["global_step"] + 1}. Its pre-step raw/EMA weights, optimiser, '
            'scaler, RNG, cursor and exact source/noise/timestep draw are retained.', '',
            '| Control | Outcome | AMP retries | Loss | Gradient norm |', '|---|---|---|---|---|',
            f'| Original policy replay | {"failed again" if replay["original_control"]["failed"] else "did not fail again"} | 0 | unavailable | nonfinite at original failure |']
        for mode in ('retry_amp', 'direct_safe_amp', 'fp32'):
            row = replay.get(mode, {})
            if row.get('finite'):
                lines.append(f'| {mode} | finite applied update | {row["amp_retries"]} | {row["loss"]:.8f} | {row["grad_norm"]:.8f} |')
            else:
                lines.append(f'| {mode} | {row.get("error", "not run")} | unavailable | unavailable | unavailable |')
        if remedy:
            lines += ['', 'Bounded AMP scale backoff recovers this draw. Its raw weights, optimiser update '
                'and post-forward RNG match a direct step at the accepted safe AMP scale bitwise. '
                'The same source/noise/timestep draw also produces a finite FP32 update. '
                'This supports AMP scale overflow as the cause of this recovered failure and validates '
                'the retry mechanism on that state. It does not prove a complete training run will be stable.']
        else:
            lines += ['', 'The complete controlled-repair gate did not pass. Keep the numerical issue '
                'unresolved and inspect the retained raw controls before planning further training.']
    else:
        lines += ['The replay reached the bounded block endpoint without recovering a failure. '
            'The historical numerical issue remains unresolved; absence of failure in this diagnostic '
            'is not proof of a fix. No artificial overflow was substituted for the missing failure.']
    ledger = latest_ledger()
    json_write(output / 'budget.json', ledger)
    lines += ['', '## Existing-data support and next work', '',
        f'Checking all existing validation slices still yields '
        f'{support_result["any_existing_fs_forward_eligible_patients"]} eligible 1.5T-to-3T patient, '
        'below the unchanged minimum of two. Resampling slices cannot repair this patient-support deficit.', '',
        'Before future efficacy continuation, freeze an explicit precision-only checkpoint migration '
        'or a new matched run: the strict resume loader correctly rejects silently changing the old '
        'config/protocol. Preserve raw/EMA/optimiser/scaler/RNG/cursor state and common training exposure, '
        'and retain fatal handling for persistent nonfinite gradients or loss. No backbone is promoted '
        'from these diagnostic updates. Independent anatomy assessment and compatible additional '
        'patient support remain separate requirements.', '',
        f'Final conservative accounting: US${ledger["estimated_prior_usd"]:.2f} including rate margin '
        f'and shared reserve, against the original approved US$6; estimated rounded remainder '
        f'US${ledger["remaining_usd"]:.2f}. Exact invoice unknown. No allowance was reset.', '',
        f'Raw evidence: `{home.relative_to(ROOT).as_posix()}/`. '
        f'Support audit: `{support.relative_to(ROOT).as_posix()}/`.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, diagnostic_analysis_completed=True,
        numerical_failure_recovered=recovered, controlled_retry_gate_passed=remedy,
        full_training_completed=False, converged=False, anatomy_validated=False,
        historical_exact_draw_identity_established=False, new_efficacy_checkpoint=False))
    print('NUMERICAL DIAGNOSTIC ANALYSED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetched', required=True)
    parser.add_argument('--support', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    analyse(args.fetched, args.support, args.run_id)
