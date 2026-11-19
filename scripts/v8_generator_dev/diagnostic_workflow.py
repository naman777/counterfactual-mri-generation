"""Prepare, launch and retrieve a single bounded numerical diagnostic, append-only."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from scripts.v8_generator_dev.failed_pair import allowance, select_failed_pair
from scripts.v8_generator_dev.fetch_full import download
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs

BASE = ROOT / 'reports/v8_generator_dev'


def latest_ledger():
    original = json.loads((BASE / 'failed_pair_report_20261117_01/budget.json').read_text())
    launches = sorted(BASE.glob('diagnostic_launch_*'))
    durations = list(original['launch_seconds'])
    for folder in launches:
        receipt = folder / 'process_result.json'
        if not receipt.is_file():
            raise ValueError('Earlier diagnostic has no terminal receipt')
        durations.append(json.loads(receipt.read_text())['seconds'])
    return allowance(durations, original['failure_seconds'])


def safe_id(value):
    if not re.fullmatch('[A-Za-z0-9_-]+', value):
        raise ValueError('Safe new workflow ID required')


def prepare(run_id, report_id):
    safe_id(run_id)
    safe_id(report_id)
    output = BASE / report_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Numerical diagnostic preparation\n\nProducer: `python -m scripts.v8_generator_dev.diagnostic_workflow prepare`.\n'
        'plan.json freezes the numerical-only scope, original checkpoint and budget; tests.txt/tests.json '
        'retain actual regression checks. code/ snapshots new diagnostic/retry/launcher sources and '
        'inputs.json hashes the archive, failure evidence, checkpoint and source. complete.json means '
        'local preparation only; it does not launch or complete generator training.\n', encoding='utf-8')
    training = BASE / 'cloud_resume_fetch_20261116_01'
    failure_path = training / 'root/failed.json'
    selection = select_failed_pair(training / 'training', json.loads(failure_path.read_text()), (training / 'root/train.txt').read_text())
    ledger = latest_ledger()
    if ledger['remaining_usd'] < .85:
        raise ValueError('Insufficient original allowance for capped diagnostic and headroom')
    bundle_receipt = BASE / 'full_bundle_20261116_01/complete.json'
    bundle = json.loads(bundle_receipt.read_text())
    archive = project_path(bundle['archive'])
    if digest(archive) != bundle['archive_sha256']:
        raise ValueError('Approved archive changed')
    modules = ['test_pipeline', 'test_full', 'test_analysis', 'test_guides', 'test_resource_limited',
               'test_verification', 'test_failed_pair', 'test_precision_step', 'test_numerical_replay']
    command = [sys.executable, '-m', 'unittest', *['scripts.v8_generator_dev.'+name for name in modules], '-v']
    with (output / 'tests.txt').open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
    json_write(output / 'tests.json', dict(command=command, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError('Diagnostic preflight failed; preserve its evidence')
    names = ['numerical_replay.py', 'precision_step.py', 'modal_numerical_replay.py', 'diagnostic_workflow.py', 'test_numerical_replay.py', 'failed_pair.py']
    inputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in (failure_path, training / 'root/train.txt', bundle_receipt, archive,
        BASE / 'failed_pair_report_20261117_01/budget.json')}
    (output / 'code').mkdir()
    for name in names:
        source = ROOT / 'scripts/v8_generator_dev' / name
        shutil.copyfile(source, output / 'code' / name)
        inputs[source.relative_to(ROOT).as_posix()] = digest(source)
    retrieval = json.loads((BASE / 'failed_pair_checkpoints_20261116_01/complete.json').read_text())
    path = project_path(retrieval['checkpoints']['additive'])
    if digest(path) != selection['checkpoint_record']['checkpoint_hashes']['additive']:
        raise ValueError('Local diagnostic checkpoint changed')
    inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    plan = dict(run_id=run_id, original_run='full_20261116_04', selection=selection,
        archive=str(archive.relative_to(ROOT)), archive_sha256=bundle['archive_sha256'],
        prior_allowance=ledger, original_approval_usd=6, new_allowance=False,
        diagnostic_allocation_usd=.60, maximum_worker_seconds=900, child_wall_seconds=720,
        pricing_source='https://modal.com/pricing', pricing_verified_date='2026-11-17',
        conservative_rate_with_margin=ledger['rate_with_margin'],
        worst_case_worker_estimate_plus_extra_reserve_usd=900*ledger['rate_with_margin']+.15,
        scope='Numerical reproduction/control updates only; never matched efficacy continuation or checkpoint promotion',
        authorised_as_small_next_step_under_original_remaining_allowance=True,
        original_engine_from_immutable_archive=True, no_test_pixels=True, no_judge_gradients=True,
        diagnostic_code_hashes={name: digest(output / 'code' / name) for name in names})
    json_write(output / 'plan.json', plan)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, local_preflight_passed=True,
        diagnostic_launched=False, failure_reproduced=False, full_training_completed=False))
    print('DIAGNOSTIC PREPARED', output, flush=True)


def launch(prepared, report_id):
    safe_id(report_id)
    prepared = project_path(prepared)
    if not json.loads((prepared / 'complete.json').read_text())['local_preflight_passed']:
        raise ValueError('Passed preflight required')
    verify_inputs(json.loads((prepared / 'inputs.json').read_text()))
    plan = json.loads((prepared / 'plan.json').read_text())
    if latest_ledger() != plan['prior_allowance']:
        raise ValueError('Remaining allowance changed since preparation')
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1',
        CAPSTONE_V8_BUNDLE=str(project_path(plan['archive'])), CAPSTONE_V8_DIAG_PLAN=str(prepared / 'plan.json'))
    inventory = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--profile', 'namankundra7', '--json'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True, env=environment)
    apps = json.loads(inventory.stdout)
    if any(app['state'] != 'stopped' for app in apps):
        raise ValueError('Another Modal app is active')
    output = BASE / report_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Numerical diagnostic launch\n\nProducer: `python -m scripts.v8_generator_dev.diagnostic_workflow launch`.\n'
        'plan.json freezes the prepared bounded job; apps_before.json checks one-app isolation; command.json '
        'and launch.log retain exact invocation/output; process_result.json retains CLI runtime/exit. '
        'No completed efficacy training or new spending allowance is implied.\n', encoding='utf-8')
    json_write(output / 'apps_before.json', apps)
    json_write(output / 'plan.json', plan)
    command = [sys.executable, '-m', 'modal', 'run', '--profile', 'namankundra7', 'scripts/v8_generator_dev/modal_numerical_replay.py']
    json_write(output / 'command.json', dict(command=command, prepared=str(prepared.relative_to(ROOT)),
        diagnostic_allocation_usd=.60, original_approval_usd=6, new_allowance=False))
    started = time.monotonic()
    with (output / 'launch.log').open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    json_write(output / 'process_result.json', dict(exit_code=result.returncode, seconds=time.monotonic()-started,
        run_id=plan['run_id'], exact_bill_known=False))
    print('DIAGNOSTIC CLI EXIT', result.returncode, flush=True)
    return result.returncode


def fetch(run_id, report_id):
    safe_id(run_id)
    safe_id(report_id)
    output = BASE / report_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Retrieved numerical diagnostic\n\nProducer: `python -m scripts.v8_generator_dev.diagnostic_workflow fetch`.\n'
        'remote/ preserves the complete worker diagnostic, snapshots and controls without changing old evidence. '
        'manifest.json maps the remote folder; inputs.json hashes every downloaded file. '
        'complete.json distinguishes diagnostic completion from efficacy training.\n', encoding='utf-8')
    record = download('v8_generator_dev/'+run_id, output / 'remote', directory=True)
    json_write(output / 'manifest.json', record)
    home = project_path(record['content_root'])
    inputs = {path.relative_to(ROOT).as_posix(): digest(path) for path in home.rglob('*') if path.is_file()}
    json_write(output / 'inputs.json', inputs)
    receipt = json.loads((home / 'complete.json').read_text()) if (home / 'complete.json').exists() else {}
    json_write(output / 'complete.json', dict(complete=bool(receipt.get('complete')), home=str(home.relative_to(ROOT)),
        full_training_completed=False, numerical_diagnostic_only=True))
    print('DIAGNOSTIC FETCHED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'fetch'])
    parser.add_argument('--run-id')
    parser.add_argument('--report-id', required=True)
    parser.add_argument('--prepared')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run_id, args.report_id)
    elif args.action == 'launch':
        raise SystemExit(launch(args.prepared, args.report_id))
    else:
        fetch(args.run_id, args.report_id)
