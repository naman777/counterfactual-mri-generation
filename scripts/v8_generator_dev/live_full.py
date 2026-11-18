"""Read current training files from the existing worker without creating an app."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

from scripts.v8_generator_dev.io import ROOT, json_write


def observe(run_id, container_id, report_id):
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (run_id, container_id, report_id)):
        raise ValueError('Invalid live observer IDs')
    output = ROOT / 'reports/v8_generator_dev' / report_id
    if output.exists():
        raise FileExistsError('Use a new observation ID')
    output.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(run_id=run_id, container_id=container_id,
                                              read_only=True, creates_app=False))
    (output / 'README.md').write_text(
        '# Live existing-worker observations\n\nProducer: `python -m scripts.v8_generator_dev.live_full`.\n'
        'protocol.json records the target worker. reader.py is the exact read-only command. '
        'stdout_*.txt/stderr_*.txt retain CLI transport output; snapshot_*.json stores '
        'actual log tails and JSON validation/phase records read from the running worker. '
        'This bypasses delayed volume visibility without altering training or creating an app.\n', encoding='utf-8')
    reader = '''import json
from pathlib import Path
home = Path('/data/v8_generator_dev/RUN_ID')
training = home / 'project/reports/v8_generator_dev/RUN_ID_training'
def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
records = []
for path in sorted(training.glob('block_*/*_validation.json')):
    value = read(path)
    if value is not None:
        records.append(dict(path=str(path.relative_to(home)), value=value))
log = home / 'train.txt'
tail = log.read_text(errors='replace').splitlines()[-10:] if log.exists() else []
print(json.dumps(dict(validation_records=records, training_tail=tail,
                      complete=read(home/'complete.json'), failed=read(home/'failed.json'))))
'''.replace('RUN_ID', run_id)
    with (output / 'reader.py').open('x', encoding='utf-8') as handle:
        handle.write(reader)
    previous = None
    for index in range(1, 401):
        command = [sys.executable, '-m', 'modal', 'container', 'exec', '--profile', 'namankundra7',
                   '--no-pty', container_id, '--', 'python', '-c', reader]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=45,
                                env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
        for name, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
            with (output / f'{name}_{index:04d}.txt').open('x', encoding='utf-8') as handle:
                handle.write(value)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            json_write(output / f'transport_{index:04d}.json', dict(exit_code=result.returncode, parsed=False))
            if result.returncode:
                print('LIVE TRANSPORT ENDED', result.returncode, flush=True)
                return
        else:
            json_write(output / f'snapshot_{index:04d}.json', dict(unix_time=time.time(), **payload))
            signature = (payload['training_tail'][-1:] or [], len(payload['validation_records']))
            if signature != previous:
                latest = payload['validation_records'][-2:]
                print('LIVE', signature[0], 'validation_records', signature[1], latest, flush=True)
                previous = signature
            if payload['complete'] is not None or payload['failed'] is not None:
                return
        time.sleep(35)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--container-id', required=True)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    observe(args.run_id, args.container_id, args.report_id)
