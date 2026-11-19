"""Read-only CLI/volume monitor with append-only snapshots and no new Modal app."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

from scripts.v8_generator_dev.io import ROOT, json_write


def cli(arguments):
    return subprocess.run([sys.executable, '-m', 'modal', *arguments, '--profile', 'namankundra7'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=45, env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))


def monitor(run_id, app_id, report_id):
    if not all(re.fullmatch('[A-Za-z0-9_-]+', x) for x in (run_id, app_id, report_id)):
        raise ValueError('Invalid monitor identifiers')
    output = ROOT / 'reports/v8_generator_dev' / report_id
    if output.exists():
        raise FileExistsError('Use a new monitor receipt ID')
    output.mkdir(parents=True)
    (output / 'README.md').write_text(
        '# Read-only Modal workflow monitoring\n\nProducer: `python -m scripts.v8_generator_dev.monitor_full`.\n'
        'protocol.json identifies the approved run/app. poll_*.json records app/volume state, '
        'remote_*.json retains completed phase/root receipts, log_*.txt retains changed phase logs, '
        'and complete.json records the observed terminal state. No app is created by monitoring.\n', encoding='utf-8')
    json_write(output / 'protocol.json', dict(run_id=run_id, app_id=app_id, read_only=True,
                                              remote_folder=f'v8_generator_dev/{run_id}'))
    seen, logs, last_state = set(), {}, None
    remote = f'v8_generator_dev/{run_id}'
    for poll in range(1, 601):
        apps = cli(['app', 'list', '--json'])
        inventory = json.loads(apps.stdout) if apps.returncode == 0 else []
        target = next((app for app in inventory if app.get('app_id') == app_id), None)
        listing = cli(['volume', 'ls', '--json', 'capstone-judge-data', remote])
        entries = json.loads(listing.stdout) if listing.returncode == 0 else []
        names = []
        for entry in entries:
            name = str(entry.get('path', entry.get('name', entry.get('filename', '')))).replace('\\', '/').rsplit('/', 1)[-1]
            if name:
                names.append(name)
        json_write(output / f'poll_{poll:04d}.json', dict(app=target, listing_exit=listing.returncode,
                                                        entries=entries, observed_names=names,
                                                        unix_time=time.time()))
        state = str(target.get('state')) if target else 'not_listed'
        if state != last_state:
            print('APP STATE', state, flush=True)
            last_state = state
        for name in sorted(names):
            if name.endswith('.json') and name not in seen:
                destination = output / f'download_{poll:04d}_{name}'
                fetched = cli(['volume', 'get', 'capstone-judge-data', f'{remote}/{name}', str(destination)])
                if fetched.returncode == 0:
                    payload = json.loads(destination.read_text(encoding='utf-8'))
                    json_write(output / f'remote_{name}', payload)
                    seen.add(name)
                    print('REMOTE RECEIPT', name, payload.get('exit_code', payload.get('complete', 'recorded')), flush=True)
            elif name.endswith('.txt'):
                destination = output / f'download_{poll:04d}_{name}'
                fetched = cli(['volume', 'get', 'capstone-judge-data', f'{remote}/{name}', str(destination)])
                value = destination.read_text(encoding='utf-8', errors='replace') if fetched.returncode == 0 else None
                if value is not None and logs.get(name) != value:
                    logs[name] = value
                    with (output / f'log_{poll:04d}_{name}').open('x', encoding='utf-8') as handle:
                        handle.write(value)
                    print('LOG', name, '\n'.join(value.splitlines()[-3:]), flush=True)
        if 'complete.json' in seen or 'failed.json' in seen:
            json_write(output / 'complete.json', dict(complete=True, remote_terminal_receipt=True,
                                                       remote_success='complete.json' in seen, app=target))
            return
        if state == 'stopped':
            json_write(output / 'complete.json', dict(complete=True, remote_terminal_receipt=False,
                                                       remote_success=False, app=target,
                                                       reason='app_stopped_without_root_completion_receipt'))
            return
        time.sleep(20)
    raise TimeoutError('Monitor duration exhausted; inspect the existing app rather than launching another')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--app-id', required=True)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    monitor(args.run_id, args.app_id, args.report_id)
