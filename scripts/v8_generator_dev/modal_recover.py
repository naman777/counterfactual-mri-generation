"""Optional remaining-budget validation after a confirmed training resource stop."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App('capstone-v8-generator-resource-validation')
volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1', 'torchvision==0.20.1', 'numpy==2.1.3', 'pandas==2.2.3',
    'scipy==1.14.1', 'scikit-image==0.24.0', 'pyyaml==6.0.2', 'pillow',
).env({'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
if modal.is_local():
    helper = Path(__file__).with_name('resource_limited.py')
    image = image.add_local_file(helper, '/root/resource_limited.py', copy=True)


@app.function(image=image, gpu='A10G', cpu=2, memory=16384, max_containers=1,
              retries=0, scaledown_window=2, timeout=7200, volumes={'/data': volume})
def recover(original_run: str, recovery_id: str, remaining_usd: float):
    import os
    from resource_limited import choose_common, remaining_allowance
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (original_run, recovery_id)):
        raise ValueError('Safe new workflow IDs required')
    started = time.monotonic()
    volume.reload()
    original = Path('/data/v8_generator_dev') / original_run
    failure = json.loads((original / 'failed.json').read_text())
    ledger = remaining_allowance(failure)
    if not 0.25 < remaining_usd <= ledger['remaining_usd']:
        raise ValueError('Recovery cannot exceed the remainder of the one approved US$6 allowance')
    project = original / 'project'
    sys.path.insert(0, str(project))
    from scripts.v8_generator_dev.io import digest, json_write, project_path
    training = project / f'reports/v8_generator_dev/{original_run}_training'
    selection = choose_common(training)
    prepared = project / f'reports/v8_generator_dev/{original_run}_protocol'
    row = selection['checkpoint_record']
    if row['prepared_sha256'] != digest(prepared / 'protocol.json'):
        raise ValueError('Common checkpoint belongs to another frozen protocol')
    for variant, relative in row['checkpoints'].items():
        path = project_path(relative)
        if digest(path) != row['checkpoint_hashes'][variant]:
            raise ValueError('Common checkpoint hash changed')
    home = Path('/data/v8_generator_dev') / recovery_id
    if home.exists():
        raise FileExistsError('Recovery outputs are append-only')
    home.mkdir()
    json_write(home / 'protocol.json', dict(original_run=original_run, recovery_id=recovery_id,
                                            allowance_ledger=ledger, allowance_used_for_recovery=remaining_usd,
                                            selection=selection,
                                            interpretation='Validation of resource-limited checkpoints, not completed/converged full training'))
    json_write(home / 'selected_common.json', selection)
    deadline = started + min(7000, (remaining_usd - 0.15) / ((0.000306 + 2 * 0.0000131 + 16 * 0.00000222) * 1.2))
    completed = []
    try:
        for variant, checkpoint in row['checkpoints'].items():
            remaining = int(deadline - time.monotonic())
            if remaining <= 30:
                raise TimeoutError('Shared approved allowance exhausted before completing recovery validation')
            command = [sys.executable, '-m', 'scripts.v8_generator_dev.evaluate_full',
                       '--prepared', str(prepared.relative_to(project)), '--checkpoint', checkpoint,
                       '--run-id', f'{recovery_id}_validation_{variant}']
            with (home / f'evaluate_{variant}.txt').open('x') as handle:
                result = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONPATH=str(project)), timeout=remaining - 15)
            json_write(home / f'evaluate_{variant}.json', dict(command=command, exit_code=result.returncode))
            volume.commit()
            if result.returncode:
                raise RuntimeError('Resource-limited validation failed: ' + variant)
            completed.append(variant)
        json_write(home / 'complete.json', dict(complete=True, validation_completed=completed,
                                                common_epoch=selection['common_epoch'], full_training_completed=False,
                                                converged=False, anatomy_validated=False,
                                                original_run=original_run, seconds=time.monotonic() - started,
                                                remaining_allowance_used=remaining_usd))
    except Exception as error:
        json_write(home / 'failed.json', dict(complete=False, error=str(error), completed_variants=completed,
                                              seconds=time.monotonic() - started, full_training_completed=False))
        raise
    finally:
        volume.commit()
    return str(home)


@app.local_entrypoint()
def main(original_run: str, recovery_id: str, remaining_usd: float):
    print(recover.remote(original_run, recovery_id, remaining_usd))
