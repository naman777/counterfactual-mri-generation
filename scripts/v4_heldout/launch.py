"""Guarded local launch or dependency queue; default is read-only preflight.

python -m scripts.v4_heldout.launch --run-id b2_20261016_01
After explicit user approval only:
python -m scripts.v4_heldout.launch --run-id b2_20261016_01 --execute --wait --approved-budget-usd 5
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.v4_convergence.core import digest, json_write
from scripts.v4_heldout.prepare_run import ROOT, valid_id


def active_apps():
    response = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--json'], capture_output=True, text=True, check=True)
    apps = json.loads(response.stdout)
    return [row for row in apps if str(row.get('state', '')).lower() != 'stopped']


def main(args):
    valid_id(args.run_id)
    base = ROOT / 'reports/v4_heldout_judge' / args.run_id
    preflight = json.loads((base / 'preflight.json').read_text())
    if not preflight['passed']:
        raise ValueError('Preflight did not pass')
    manifest = json.loads((base / 'source_manifest.json').read_text())
    for row in manifest:
        if digest(ROOT / row['path']) != row['sha256']:
            raise ValueError(f'Sealed source changed: {row["path"]}')
    apps = active_apps()
    print(json.dumps({'run_id': args.run_id, 'execute': args.execute, 'active_apps': [a['app_id'] for a in apps], 'status': 'waiting_for_existing_app' if apps else 'ready', 'requested_total_allowance': args.approved_budget_usd}, indent=2), flush=True)
    if not args.execute:
        return
    if args.approved_budget_usd is None or args.approved_budget_usd < .5:
        raise ValueError('Supply the separately approved total allowance')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    json_write(base / 'queue' / f'{attempt}_request.json', {'explicit_compute_approval_supplied': True, 'total_allowance_usd': args.approved_budget_usd, 'setup_reserve_usd': .25, 'controller_cap_usd': args.approved_budget_usd - .25, 'wait_for_other_apps': args.wait, 'requires_local_machine_online_until_dispatch': True})
    deadline = time.monotonic() + 8 * 3600
    last = [a['app_id'] for a in apps]
    while apps:
        if not args.wait:
            raise RuntimeError('Another Modal app is active; no new app created')
        if time.monotonic() > deadline:
            raise TimeoutError('Existing app still active after eight hours; no cloud launch')
        time.sleep(30)
        apps = active_apps()
        current = [a['app_id'] for a in apps]
        if current != last:
            print('Active apps:', current, flush=True)
            last = current
    # No paid worker or new app exists before this point.
    command = [sys.executable, '-m', 'modal', 'run', '--detach', 'scripts/v4_heldout/modal_app.py', '--run-id', args.run_id, '--budget-usd', str(args.approved_budget_usd - .25)]
    json_write(base / 'queue' / f'{attempt}_dispatch.json', {'command': command, 'active_app_check': [], 'dispatch_utc': datetime.now(timezone.utc).isoformat()})
    result = subprocess.run(command, cwd=ROOT, check=False)
    json_write(base / 'queue' / f'{attempt}_exit.json', {'returncode': result.returncode})
    if result.returncode:
        raise RuntimeError('Modal launch/controller process failed; see preserved console log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--approved-budget-usd', type=float)
    main(parser.parse_args())
