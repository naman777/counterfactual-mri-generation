"""Launch validation only within the remaining allowance; retain every receipt."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from scripts.v8_generator_dev.failed_pair import allowance, select_failed_pair
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def launch(fetched, recovery_id, report_id):
    import re
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (recovery_id, report_id)):
        raise ValueError('Safe new IDs required')
    fetched = project_path(fetched)
    original_run = json.loads((fetched / 'protocol.json').read_text())['run_id']
    failure = json.loads((fetched / 'root/failed.json').read_text())
    selection = select_failed_pair(fetched / 'training', failure, (fetched / 'root/train.txt').read_text())
    launches = sorted((ROOT / 'reports/v8_generator_dev').glob('cloud_launch_20261116_*'))
    if len(launches) != 4:
        raise ValueError('Account for all four original dispatches before recovery')
    durations, receipts, missing = [], [], []
    for folder in launches:
        receipt = folder / 'process_result.json'
        if receipt.exists():
            durations.append(json.loads(receipt.read_text())['seconds'])
            receipts.append(receipt)
        else:
            durations.append(600.0)
            missing.append(str(folder.relative_to(ROOT)))
            receipts.append(folder / 'launch.log')
    prior_recoveries = sorted((ROOT / 'reports/v8_generator_dev').glob('failed_pair_launch_*'))
    for folder in prior_recoveries:
        receipt = folder / 'process_result.json'
        if receipt.exists():
            durations.append(json.loads(receipt.read_text())['seconds'])
            receipts.append(receipt)
        else:
            raise ValueError('An earlier validation launch lacks a terminal duration receipt')
    ledger = allowance(durations, failure['seconds'])
    if ledger['remaining_usd'] <= 0.25:
        raise ValueError('Original allowance remainder is insufficient')
    bundle_receipt = ROOT / 'reports/v8_generator_dev/full_bundle_20261116_01/complete.json'
    bundle = json.loads(bundle_receipt.read_text())
    archive = project_path(bundle['archive'])
    if digest(archive) != bundle['archive_sha256']:
        raise ValueError('Original approved input archive changed')
    environment_bundle = str(archive)
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1',
                       CAPSTONE_V8_BUNDLE=environment_bundle)
    apps = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--profile', 'namankundra7', '--json'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True, env=environment)
    inventory = json.loads(apps.stdout)
    if any(row['state'] != 'stopped' for row in inventory):
        raise ValueError('Another Modal app is active')
    output = ROOT / 'reports/v8_generator_dev' / report_id
    output.mkdir(exist_ok=False)
    json_write(output / 'allowance.json', ledger)
    json_write(output / 'duration_evidence.json', dict(missing_duration_receipts=missing,
                                                      conservative_seconds_per_missing_dispatch=600,
                                                      exact_bill_known=False))
    json_write(output / 'selected_common.json', selection)
    json_write(output / 'apps_before.json', inventory)
    json_write(output / 'inputs.json', {path.relative_to(ROOT).as_posix(): digest(path) for path in receipts})
    entrypoint = ROOT / 'scripts/v8_generator_dev/modal_failed_validation.py'
    shutil.copyfile(entrypoint, output / 'entrypoint_snapshot.py')
    shutil.copyfile(entrypoint.with_name('failed_pair.py'), output / 'selector_snapshot.py')
    command = [sys.executable, '-m', 'modal', 'run', '--profile', 'namankundra7',
               str(entrypoint.relative_to(ROOT)), '--original-run', original_run,
               '--recovery-id', recovery_id, '--ledger-json', json.dumps(ledger)]
    json_write(output / 'command.json', dict(command=command, recovery_id=recovery_id,
                                             original_run=original_run, training_restarted=False,
                                             original_approval_usd=6, new_allowance=False))
    (output / 'README.md').write_text(
        '# Numerical-failure validation launch\n\nProducer: `python -m scripts.v8_generator_dev.launch_failed_validation`.\n'
        'allowance.json debits all four original launches; inputs.json hashes their receipts. '
        'selected_common.json retains the last matched pair and incomplete training status. '
        'apps_before.json checks single-app isolation; command.json and *_snapshot.py freeze execution. '
        'launch.log/process_result.json retain CLI output/exit, not scientific completion.\n', encoding='utf-8')
    started = time.perf_counter()
    print('VALIDATION ONLY', original_run, selection['common_epoch'], ledger, flush=True)
    with (output / 'launch.log').open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    json_write(output / 'process_result.json', dict(exit_code=result.returncode,
                                                   seconds=time.perf_counter() - started, recovery_id=recovery_id))
    print('VALIDATION CLI EXIT', result.returncode, flush=True)
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetched', required=True)
    parser.add_argument('--recovery-id', required=True)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.fetched, args.recovery_id, args.report_id))
