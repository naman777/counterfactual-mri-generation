"""Launch the explicitly approved private Modal workflow and retain its CLI log."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def launch(bundle_folder, run_id, report_id, approved_usd):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (run_id, report_id)):
        raise ValueError('New safe run/report IDs required')
    if approved_usd != 6.0:
        raise ValueError('This launch records the specific US$6 user approval')
    bundle_folder = project_path(bundle_folder)
    bundle = json.loads((bundle_folder / 'complete.json').read_text())
    archive = project_path(bundle['archive'])
    if not bundle['local_archive_verified'] or digest(archive) != bundle['archive_sha256']:
        raise ValueError('Verified archive hash changed')
    output = ROOT / 'reports/v8_generator_dev' / report_id
    if output.exists():
        raise FileExistsError('Launch receipts are append-only')
    output.mkdir(parents=True)
    apps = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--profile', 'namankundra7', '--json'],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    inventory = json.loads(apps.stdout)
    json_write(output / 'apps_before.json', inventory)
    active = [app for app in inventory if str(app.get('state', '')).lower() != 'stopped']
    if active:
        raise RuntimeError('An existing Modal app needs review before this one-app workflow launches')
    json_write(output / 'approval.json', dict(
        user_approved_usd=approved_usd, scope='Private input upload, one-A10 sequential R0/R1 training and validation',
        source='Explicit user reply: Approve US$6', client_date='2026-11-16', run_id=run_id,
        bundle=str(archive.relative_to(ROOT)), bundle_sha256=bundle['archive_sha256'],
        pricing_proposal=str((bundle_folder / 'execution_proposal.json').relative_to(ROOT)),
    ))
    command = [sys.executable, '-m', 'modal', 'run', '--profile', 'namankundra7',
               'scripts/v8_generator_dev/modal_full.py', '--run-id', run_id, '--budget-usd', str(approved_usd)]
    json_write(output / 'command.json', dict(command=command, cwd=str(ROOT),
                                             environment_override={'CAPSTONE_V8_BUNDLE': str(archive)},
                                             entrypoint_sha256=digest(ROOT / 'scripts/v8_generator_dev/modal_full.py')))
    shutil.copyfile(ROOT / 'scripts/v8_generator_dev/modal_full.py', output / 'modal_full_snapshot.py')
    (output / 'README.md').write_text(
        '# Approved Modal launch receipt\n\nProducer: `python -m scripts.v8_generator_dev.launch_full`.\n'
        'approval.json records the explicit user allowance and archive hash. apps_before.json checks '
        'single-app isolation. command.json and modal_full_snapshot.py retain the exact entrypoint. '
        'launch.log is the actual CLI build/upload/run output. process_result.json records its exit. '
        'Further remote-state/fetch receipts use new files/folders; CLI success alone is not scientific completion.\n',
        encoding='utf-8')
    environment = dict(os.environ, CAPSTONE_V8_BUNDLE=str(archive), PYTHONUNBUFFERED='1',
                       PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    start = time.perf_counter()
    print('LAUNCHING', run_id, 'approved compute US$', approved_usd, flush=True)
    with (output / 'launch.log').open('x', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if any(word in line.lower() for word in ('view run', 'initialized', 'building image', 'finished',
                                                     'created', 'traceback', 'error', 'exception', 'stopping')):
                print(line.rstrip(), flush=True)
        code = process.wait()
    json_write(output / 'process_result.json', dict(exit_code=code, seconds=time.perf_counter() - start,
                                                   run_id=run_id, remote_folder=f'/data/v8_generator_dev/{run_id}',
                                                   scientific_completion_requires_remote_receipt=True))
    print('MODAL CLI EXIT', code, 'receipt', output, flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-folder', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-id', required=True)
    parser.add_argument('--approved-usd', type=float, required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.bundle_folder, args.run_id, args.report_id, args.approved_usd))
