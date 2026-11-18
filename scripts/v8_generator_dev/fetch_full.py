"""Fetch this approved workflow's reports and only its final primary checkpoints."""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def command(arguments):
    return subprocess.run([sys.executable, '-m', 'modal', 'volume', *arguments, '--profile', 'namankundra7'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'), timeout=1800)


def download(remote, destination, directory=False):
    if destination.exists():
        raise FileExistsError('Never replace a local output: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        destination.mkdir()
    result = command(['get', 'capstone-judge-data', remote, str(destination)])
    if result.returncode:
        raise RuntimeError(f'Fetch failed for {remote}: {result.stderr[-500:]}')
    return dict(remote=remote, local=str(destination.relative_to(ROOT)), exit_code=result.returncode,
                content_root=str((destination / remote.rsplit('/', 1)[-1]).relative_to(ROOT)) if directory else None,
                sha256=digest(destination) if destination.is_file() else None)


def fetch(run_id, report_id):
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (run_id, report_id)):
        raise ValueError('Invalid fetch IDs')
    remote = f'v8_generator_dev/{run_id}'
    output = ROOT / 'reports/v8_generator_dev' / report_id
    if output.exists():
        raise FileExistsError('Use a new fetch receipt ID')
    output.mkdir(parents=True)
    (output / 'README.md').write_text(
        '# Approved workflow result retrieval\n\nProducer: `python -m scripts.v8_generator_dev.fetch_full`.\n'
        'protocol.json records remote scope. root/ retains original phase/root receipts and logs. '
        'protocol/ is the authorised frozen remote protocol. training/ retains completion/input files '
        'and per-block validation/paired receipts; raw draw logs remain on the volume. '
        'validation_*/ retains every output from each completed evaluation. fetch_manifest.json '
        'and inputs.json bind downloaded bytes; complete.json records full completion versus failure. '
        'Only final primary checkpoints are fetched, into new checkpoint paths, after hash checks. '
        'No original dataset, historical result or model is overwritten.\n', encoding='utf-8')
    json_write(output / 'protocol.json', dict(run_id=run_id, remote_folder=remote,
                                              retrieval='Reports and final primary checkpoints only; no bulk data/intermediate checkpoint download'))
    listing = command(['ls', '--json', 'capstone-judge-data', remote])
    if listing.returncode:
        raise RuntimeError('Remote workflow folder is unavailable')
    entries = json.loads(listing.stdout)
    records = []
    for entry in entries:
        name = entry['filename'].rsplit('/', 1)[-1]
        if entry['type'] == 'file':
            records.append(download(f'{remote}/{name}', output / 'root' / name))
    project = f'{remote}/project'
    protocol = f'{project}/reports/v8_generator_dev/{run_id}_protocol'
    records.append(download(protocol, output / 'protocol', directory=True))
    training = f'{project}/reports/v8_generator_dev/{run_id}_training'
    listing = command(['ls', '--json', 'capstone-judge-data', training])
    entries = json.loads(listing.stdout) if not listing.returncode else []
    tasks = []
    for entry in entries:
        name = entry['filename'].rsplit('/', 1)[-1]
        if entry['type'] == 'file':
            tasks.append((f'{training}/{name}', output / 'training' / name))
        elif re.fullmatch(r'block_\d+', name):
            for filename in ('complete.json', 'additive_validation.json', 'scale_shift_validation.json',
                             'additive_validation.csv', 'scale_shift_validation.csv'):
                tasks.append((f'{training}/{name}/{filename}', output / 'training' / name / filename))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, *task) for task in tasks]
        for future in futures:
            try:
                records.append(future.result())
            except RuntimeError as error:
                # A resource-stop block may be incomplete; preserve explicit failure.
                records.append(dict(fetch_error=str(error)))
    root_complete = output / 'root/complete.json'
    trained_path = output / 'training/complete.json'
    complete = root_complete.exists() and trained_path.exists()
    evaluation_folders = {}
    if complete:
        trained = json.loads(trained_path.read_text())
        for variant in ('additive', 'scale_shift'):
            record = download(f'{project}/reports/v8_generator_dev/{run_id}_validation_{variant}',
                              output / f'validation_{variant}', directory=True)
            records.append(record)
            evaluation_folders[variant] = record['content_root']
        for variant, relative in trained['checkpoints'].items():
            destination = project_path(relative)
            if not destination.is_relative_to(ROOT / 'checkpoints'):
                raise ValueError('Checkpoint destination outside checkpoint directory')
            records.append(download(f'{project}/{relative.replace(chr(92), "/")}', destination))
            if digest(destination) != trained['checkpoint_hashes'][variant]:
                raise ValueError('Downloaded primary checkpoint hash mismatch')
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in output.rglob('*') if path.is_file()}
    json_write(output / 'fetch_manifest.json', records)
    json_write(output / 'inputs.json', hashes)
    json_write(output / 'complete.json', dict(
        complete=True, workflow_success=complete, reports_retrieved=True,
        primary_checkpoint_hashes_verified=bool(complete),
        full_training_completed=bool(complete), clinical_preservation='unknown',
        training_folder=str((output / 'training').relative_to(ROOT)), evaluation_folders=evaluation_folders,
        failed_workflow_receipt=(output / 'root/failed.json').exists(),
        scope='Fetch completion is not scientific or clinical success',
    ))
    print('FETCH COMPLETE', output, 'workflow_success', complete, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    fetch(args.run_id, args.report_id)
