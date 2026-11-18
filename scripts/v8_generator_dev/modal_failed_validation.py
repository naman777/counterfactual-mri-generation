"""Validate a frozen last-common pair after numerical failure; never resume training."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App('capstone-v8-failed-pair-validation')
volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1', 'torchvision==0.20.1', 'numpy==2.1.3', 'pandas==2.2.3',
    'scipy==1.14.1', 'scikit-image==0.24.0', 'pyyaml==6.0.2', 'pillow',
).env({'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
if modal.is_local():
    bundle_path = Path(os.environ['CAPSTONE_V8_BUNDLE']).resolve()
    image = image.add_local_file(bundle_path, '/input/inputs.zip', copy=True)
    image = image.add_local_file(Path(__file__).with_name('failed_pair.py'), '/root/failed_pair.py', copy=True)


@app.function(image=image, gpu='A10G', cpu=2, memory=16384, max_containers=1,
              retries=0, scaledown_window=2, timeout=7200, volumes={'/data': volume})
def validate(original_run: str, recovery_id: str, ledger_json: str):
    import hashlib
    import shutil
    import zipfile
    from pathlib import PurePosixPath
    from failed_pair import RATE_WITH_MARGIN, allowance, select_failed_pair
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (original_run, recovery_id)):
        raise ValueError('Safe workflow IDs required')
    started = time.monotonic()
    volume.reload()
    original = Path('/data/v8_generator_dev') / original_run
    failure = json.loads((original / 'failed.json').read_text())
    supplied = json.loads(ledger_json)
    ledger = allowance(supplied['launch_seconds'], failure['seconds'])
    if ledger != supplied or ledger['remaining_usd'] <= 0.25:
        raise ValueError('Invalid remaining original allowance')
    project = original / 'project'
    sys.path.insert(0, str(project))
    from scripts.v8_generator_dev.io import digest, json_write, project_path
    training = project / f'reports/v8_generator_dev/{original_run}_training'
    selection = select_failed_pair(training, failure, (original / 'train.txt').read_text())
    prepared = project / f'reports/v8_generator_dev/{original_run}_protocol'
    record = selection['checkpoint_record']
    if record['prepared_sha256'] != digest(prepared / 'protocol.json'):
        raise ValueError('Checkpoint protocol mismatch')
    for variant, relative in record['checkpoints'].items():
        if digest(project_path(relative)) != record['checkpoint_hashes'][variant]:
            raise ValueError('Checkpoint hash mismatch')
    home = Path('/data/v8_generator_dev') / recovery_id
    home.mkdir(exist_ok=False)
    json_write(home / 'protocol.json', dict(original_run=original_run, recovery_id=recovery_id,
                                            allowance_ledger=ledger, selection=selection,
                                            training_restarted=False, frozen_evaluation_unchanged=True))
    json_write(home / 'selected_common.json', selection)
    volume.commit()
    deadline = started + min(7000, (ledger['remaining_usd'] - 0.15) / RATE_WITH_MARGIN)
    completed = []
    try:
        staged = Path('/tmp') / recovery_id
        staged.mkdir(exist_ok=False)
        with zipfile.ZipFile('/input/inputs.zip') as archive:
            manifest = json.loads(archive.read('_payload_manifest.json'))
            if set(archive.namelist()) != set(manifest) | {'_payload_manifest.json'}:
                raise ValueError('Archive manifest mismatch')
            for name, expected in manifest.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe archive member')
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError('Archive input hash mismatch')
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        staged_prepared = staged / prepared.relative_to(project)
        shutil.copytree(prepared, staged_prepared)
        required = json.loads((prepared / 'inputs.json').read_text())
        required.update({record['checkpoints'][variant]: expected
                         for variant, expected in record['checkpoint_hashes'].items()})
        copied = []
        for relative, expected in required.items():
            destination = staged / str(relative).replace('\\', '/')
            if not destination.is_relative_to(staged) or '..' in destination.parts:
                raise ValueError('Staged path escapes scratch project')
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project_path(relative), destination)
                copied.append(relative)
            if digest(destination) != expected:
                raise ValueError('Staged frozen input mismatch: ' + relative)
        json_write(home / 'staging.json', dict(complete=True, seconds=time.monotonic() - started,
                                               archive_members_verified=len(manifest),
                                               frozen_inputs_verified=len(required), extra_files=copied,
                                               input_bytes_unchanged=True, scientific_code_unchanged=True))
        volume.commit()
        for variant, checkpoint in record['checkpoints'].items():
            remaining = int(deadline - time.monotonic())
            if remaining <= 30:
                raise TimeoutError('Original allowance remainder exhausted')
            command = [sys.executable, '-m', 'scripts.v8_generator_dev.evaluate_full',
                       '--prepared', str(prepared.relative_to(project)), '--checkpoint', checkpoint,
                       '--run-id', f'{recovery_id}_validation_{variant}']
            with (home / f'evaluate_{variant}.txt').open('x') as handle:
                result = subprocess.run(command, cwd=staged, stdout=handle, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONPATH=str(staged), PYTHONUNBUFFERED='1'),
                                        timeout=remaining - 15)
            evaluation = f'reports/v8_generator_dev/{recovery_id}_validation_{variant}'
            if (staged / evaluation).exists():
                shutil.copytree(staged / evaluation, project / evaluation)
            json_write(home / f'evaluate_{variant}.json', dict(command=command, exit_code=result.returncode))
            volume.commit()
            if result.returncode:
                raise RuntimeError('Validation failed: ' + variant)
            completed.append(variant)
        json_write(home / 'complete.json', dict(complete=True, validation_completed=completed,
                                                common_epoch=selection['common_epoch'], full_training_completed=False,
                                                converged=False, anatomy_validated=False, original_run=original_run,
                                                seconds=time.monotonic() - started, allowance_ledger=ledger))
    except Exception as error:
        json_write(home / 'failed.json', dict(complete=False, error=str(error), completed_variants=completed,
                                              seconds=time.monotonic() - started, full_training_completed=False))
        raise
    finally:
        volume.commit()
    return str(home)


@app.local_entrypoint()
def main(original_run: str, recovery_id: str, ledger_json: str):
    print(validate.remote(original_run, recovery_id, ledger_json))
