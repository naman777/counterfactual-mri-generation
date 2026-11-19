"""Close only this authorised allocation with provider app and timing evidence."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def record(prepared, results, launch, app_id, run_id):
    prepared, results, launch = map(project_path, (prepared, results, launch))
    plan = json.loads((prepared / 'plan.json').read_text())
    cli = json.loads((launch / 'result.json').read_text())
    terminal = results / 'root/complete.json'
    receipt = json.loads(terminal.read_text())
    command = [sys.executable, '-m', 'modal', 'app', 'list', '--profile', 'namankundra7', '--json']
    observed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    apps = json.loads(observed.stdout)
    own = [a for a in apps if a['app_id'] == app_id]
    if len(own) != 1 or own[0]['state'] != 'stopped' or own[0]['stopped_at'] is None:
        raise ValueError('Provider-observed stopped app required before closing allocation')
    app = own[0]
    elapsed = (datetime.fromisoformat(app['stopped_at']) - datetime.fromisoformat(app['created_at'])).total_seconds()
    seconds = max(elapsed, receipt['seconds'])
    estimate = seconds * plan['price_with_margin'] + plan['reserve_usd']
    output = ROOT / 'reports/v10_detail' / run_id
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / 'apps_after.json', apps)
    json_write(output / 'budget.json', dict(approved_new_usd=12, app_id=app_id,
        worker_seconds=receipt['seconds'], maximum_worker_seconds=plan['maximum_worker_seconds'],
        provider_app_elapsed_seconds=elapsed, conservative_seconds_used=seconds,
        conservative_rate_usd_per_second=plan['price_with_margin'], reserve_usd=plan['reserve_usd'],
        conservative_estimate_usd=estimate, provider_invoice_unknown=True,
        worker_time_within_cap=receipt['seconds'] <= plan['maximum_worker_seconds'],
        conservative_estimate_within_cap=estimate <= 12, cli_exit_code=cli['exit_code'],
        app_stopped=True, automatic_paid_restart=False, previous_allowances_reset=False,
        observed_utc=datetime.now(timezone.utc).isoformat()))
    json_write(output / 'inputs.json', {path.relative_to(ROOT).as_posix(): digest(path) for path in
        (project_path(__file__), prepared / 'plan.json', prepared / 'pricing.json', launch / 'result.json',
         launch / 'command.json', terminal)})
    (output / 'README.md').write_text(
        'Producer: scripts.v10_detail.record_budget. apps_after.json: provider-observed stopped app and inventory; '
        'budget.json: separate new US$12 cap, actual worker duration, conservative maximum of worker and '
        'provider app-wall duration times the frozen marked-up GPU/CPU/RAM rate, plus the frozen reserve; '
        'inputs.json: original pricing/plan/CLI/terminal evidence and producer. This is a conservative estimate, '
        'not a provider invoice; old budgets stay closed and no unused balance authorises another run.\n', encoding='utf-8')
    print('ALLOCATION CLOSED', round(estimate, 4), 'USD conservative estimate; invoice unknown', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ('prepared', 'results', 'launch', 'app-id', 'run-id'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); record(a.prepared, a.results, a.launch, a.app_id, a.run_id)
