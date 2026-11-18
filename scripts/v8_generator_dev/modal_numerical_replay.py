"""One short A10 diagnostic under the remainder of the original allowance."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import modal

app = modal.App('capstone-v8-numerical-diagnostic')
volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1', 'torchvision==0.20.1', 'numpy==2.1.3', 'pandas==2.2.3',
    'scipy==1.14.1', 'scikit-image==0.24.0', 'pyyaml==6.0.2', 'pillow',
).env({'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
if modal.is_local():
    image = image.add_local_file(Path(os.environ['CAPSTONE_V8_BUNDLE']), '/input/inputs.zip', copy=True)
    image = image.add_local_file(Path(os.environ['CAPSTONE_V8_DIAG_PLAN']), '/input/plan.json', copy=True)
    for name in ('numerical_replay.py', 'precision_step.py', 'failed_pair.py'):
        image = image.add_local_file(Path(__file__).with_name(name), '/root/' + name, copy=True)


@app.function(image=image, gpu='A10G', cpu=2, memory=16384, max_containers=1,
              retries=0, scaledown_window=2, timeout=900, volumes={'/data': volume})
def diagnose():
    from failed_pair import select_failed_pair
    started = time.monotonic()
    plan = json.loads(Path('/input/plan.json').read_text())
    if plan['maximum_worker_seconds'] != 900 or plan['diagnostic_allocation_usd'] > 0.60:
        raise ValueError('Diagnostic ceilings changed')
    if plan['prior_allowance']['remaining_usd'] < plan['diagnostic_allocation_usd'] + .25:
        raise ValueError('Insufficient original allowance remainder')
    run_id = plan['run_id']
    if not run_id.replace('_', '').isalnum():
        raise ValueError('Unsafe new ID')
    volume.reload()
    original = Path('/data/v8_generator_dev/full_20261116_04')
    project = original / 'project'
    sys.path.insert(0, str(project))
    from scripts.v8_generator_dev.io import digest, json_write
    training = project / 'reports/v8_generator_dev/full_20261116_04_training'
    selection = select_failed_pair(training, json.loads((original / 'failed.json').read_text()), (original / 'train.txt').read_text())
    if selection['checkpoint_record'] != plan['selection']['checkpoint_record']:
        raise ValueError('Diagnostic checkpoint selection changed')
    home = Path('/data/v8_generator_dev') / run_id
    home.mkdir(exist_ok=False)
    json_write(home / 'protocol.json', plan)
    volume.commit()
    staged = Path('/tmp') / run_id
    staged.mkdir(exist_ok=False)
    result_folder = staged / 'reports/v8_generator_dev' / run_id
    try:
        if digest(Path('/input/inputs.zip')) != plan['archive_sha256']:
            raise ValueError('Original archive changed')
        with zipfile.ZipFile('/input/inputs.zip') as archive:
            manifest = json.loads(archive.read('_payload_manifest.json'))
            if set(archive.namelist()) != set(manifest) | {'_payload_manifest.json'}:
                raise ValueError('Archive membership changed')
            for name, expected in manifest.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe archive member')
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError('Archive member changed')
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        prepared_relative = 'reports/v8_generator_dev/full_20261116_04_protocol'
        prepared = project / prepared_relative
        shutil.copytree(prepared, staged / prepared_relative)
        required = json.loads((prepared / 'inputs.json').read_text())
        record = selection['checkpoint_record']
        checkpoint = record['checkpoints']['additive']
        required[checkpoint] = record['checkpoint_hashes']['additive']
        for relative, expected in required.items():
            destination = staged / relative.replace('\\', '/')
            if not destination.resolve().is_relative_to(staged):
                raise ValueError('Unsafe frozen input path')
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / relative.replace('\\', '/'), destination)
            if digest(destination) != expected:
                raise ValueError('Frozen input mismatch: ' + relative)
        for name in ('numerical_replay.py', 'precision_step.py'):
            destination = staged / 'scripts/v8_generator_dev' / name
            if destination.exists():
                raise FileExistsError('Diagnostic must not replace original scientific code')
            source = Path('/root') / name
            if digest(source) != plan['diagnostic_code_hashes'][name]:
                raise ValueError('Diagnostic code mismatch')
            shutil.copyfile(source, destination)
        json_write(home / 'staging.json', dict(complete=True, seconds=time.monotonic()-started,
            frozen_input_hashes_verified=len(required), original_engine_unchanged=True,
            checkpoint_sha256=required[checkpoint], diagnostic_code_added_separately=True))
        volume.commit()
        command = [sys.executable, '-m', 'scripts.v8_generator_dev.numerical_replay',
            '--prepared', prepared_relative, '--checkpoint', checkpoint, '--run-id', run_id, '--wall-seconds', '720']
        remaining = min(750, 850 - int(time.monotonic()-started))
        if remaining < 200:
            raise TimeoutError('Staging exhausted diagnostic cap')
        with (home / 'diagnostic.txt').open('x') as handle:
            process = subprocess.run(command, cwd=staged, stdout=handle, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH=str(staged), PYTHONUNBUFFERED='1'), timeout=remaining)
        if result_folder.exists():
            shutil.copytree(result_folder, home / 'results')
        json_write(home / 'process_result.json', dict(exit_code=process.returncode, seconds=time.monotonic()-started))
        if process.returncode:
            raise RuntimeError('Diagnostic process failed; partial evidence retained')
        result = json.loads((home / 'results/complete.json').read_text())
        json_write(home / 'complete.json', dict(complete=True, diagnostic=result,
            seconds=time.monotonic()-started, full_training_completed=False, new_efficacy_checkpoint=False))
    except Exception as error:
        if result_folder.exists() and not (home / 'results').exists():
            shutil.copytree(result_folder, home / 'results')
        json_write(home / 'failed.json', dict(complete=False, error=str(error), seconds=time.monotonic()-started))
        raise
    finally:
        volume.commit()
    return str(home)


@app.local_entrypoint()
def main():
    print(diagnose.remote())
